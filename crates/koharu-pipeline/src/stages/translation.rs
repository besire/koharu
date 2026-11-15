use std::{collections::BTreeMap, str::FromStr};

use anyhow::Result;
use async_trait::async_trait;
use koharu_scene::{
    Authored, EntityId, LanguageTag, Origin, Snapshot, SourceText, TextContent, Translation,
};
use koharu_translator::{Language, TranslationContext, TranslationRequest, Translator};

use crate::{GlossaryEntry, TranslationConfig};

use super::{StageInput, StageProcessor, finish, generation};

const PRODUCER: &str = "dev.koharu.pipeline.translation";
const CONTEXT_PAGE_WINDOW: usize = 3;
const MAX_HISTORY_CONTEXT_ENTRIES: usize = 36;

pub(super) struct Processor {
    config: TranslationConfig,
    translator: Translator,
}

struct TranslationTarget {
    content: EntityId,
    source: String,
    note: Option<String>,
    language: Option<Language>,
}

impl Processor {
    pub(super) fn new(config: TranslationConfig, translator: Translator) -> Self {
        Self { config, translator }
    }
}

#[async_trait]
impl StageProcessor for Processor {
    fn model(&self) -> &'static str {
        Translator::model(&self.config.model)
    }

    fn unload(&self) -> bool {
        self.translator.unload()
    }

    async fn load(&self) -> Result<()> {
        self.translator.load_model(&self.config.model).await
    }

    async fn process(&self, input: StageInput) -> Result<koharu_scene::Patch> {
        let mut targets = Vec::new();
        if let Some(group) = input.scene.page(input.page)?.text_group()? {
            for layer in group.text_layers()? {
                if !input.contains_entity(layer.id())? {
                    continue;
                }
                let content = layer.content()?;
                let Some(source) = content.source()? else {
                    continue;
                };
                if !source.text.value.trim().is_empty() {
                    let language = source
                        .language
                        .as_ref()
                        .and_then(|tag| Language::from_str(tag.as_str()).ok());
                    targets.push(TranslationTarget {
                        content: content.id(),
                        source: source.text.value,
                        note: content.note()?,
                        language,
                    });
                }
            }
        }
        let target_ids = targets
            .iter()
            .map(|target| target.content)
            .collect::<Vec<_>>();
        let context = collect_translation_context(
            &input.scene,
            input.page,
            &self.config.glossary,
            input.has_entity_scope(),
            &target_ids,
        )?;
        let mut request = TranslationRequest::new(
            targets.iter().map(|target| target.source.clone()),
            self.config.target_language,
        )
        .with_segment_notes(targets.iter().map(|target| target.note.clone()));
        if !context.is_empty() {
            request = request.with_context(context);
        }
        if let Some(source_language) =
            dominant_source_language(&targets, self.config.target_language)
        {
            request = request.with_source_language(source_language);
        }
        if let Some(instructions) = self.config.instructions.as_deref() {
            request = request.with_instructions(instructions);
        }
        if Translator::supports_vision(&self.config.model, &self.config.generation)
            && let Some(image) = input.images.get(&input.scene, input.page, "source").await?
        {
            request = request.with_image(image);
        }
        let (provider, translated) = self
            .translator
            .translate(&self.config.model, self.config.generation, request)
            .await?;
        let language = LanguageTag::new(self.config.target_language.tag())?;
        let generated = generation(PRODUCER, provider)?;
        let overwrite_user = input.has_entity_scope();
        let mut edit = input.scene.edit_as(generated.clone());
        for target in &targets {
            edit.observe::<TextContent>(target.content)?;
            edit.observe::<SourceText>(target.content)?;
            edit.observe::<Translation>(target.content)?;
        }
        for (target, text) in targets.into_iter().zip(translated) {
            if !overwrite_user
                && input
                    .scene
                    .component::<Translation>(target.content)?
                    .is_some_and(|value| matches!(value.text.origin, Origin::User))
            {
                continue;
            }
            let text = if target.source.trim() == "\u{2026}" {
                "\u{2026}".to_owned()
            } else {
                text
            };
            edit.set_generated_translation(
                target.content,
                &Translation {
                    text: Authored::generated(text, generated.clone()),
                    language: Some(language.clone()),
                },
                overwrite_user,
            )?;
        }
        finish(edit)
    }
}

fn dominant_source_language(
    targets: &[TranslationTarget],
    target_language: Language,
) -> Option<Language> {
    let mut counts = BTreeMap::<String, (Language, usize)>::new();
    for language in targets.iter().filter_map(|target| target.language) {
        if language == target_language {
            continue;
        }
        counts
            .entry(language.tag().to_owned())
            .and_modify(|(_, count)| *count += 1)
            .or_insert((language, 1));
    }
    counts
        .into_values()
        .max_by_key(|(_, count)| *count)
        .map(|(language, _)| language)
}

fn collect_translation_context(
    scene: &Snapshot,
    current_page: EntityId,
    glossary: &[GlossaryEntry],
    include_current_page_siblings: bool,
    target_ids: &[EntityId],
) -> Result<Vec<TranslationContext>> {
    let mut context = Vec::new();
    for entry in glossary {
        let source = entry.source.trim();
        let target = entry.target.trim();
        if !source.is_empty() && !target.is_empty() {
            push_unique_context(&mut context, source, target);
        }
    }

    let pages = scene.pages().map(|page| page.id()).collect::<Vec<_>>();
    let current_index = pages
        .iter()
        .position(|&id| id == current_page)
        .unwrap_or(pages.len());

    let mut history_pages = Vec::new();
    for &page_id in pages[..current_index].iter().rev() {
        let pairs = page_translated_pairs(scene, page_id, &[])?;
        if !pairs.is_empty() {
            history_pages.push(pairs);
            if history_pages.len() >= CONTEXT_PAGE_WINDOW {
                break;
            }
        }
    }
    history_pages.reverse();

    let mut history_entries = history_pages.into_iter().flatten().collect::<Vec<_>>();
    if history_entries.len() > MAX_HISTORY_CONTEXT_ENTRIES {
        let skip = history_entries.len() - MAX_HISTORY_CONTEXT_ENTRIES;
        history_entries.drain(..skip);
    }
    for (source, translation) in history_entries {
        push_unique_context(&mut context, &source, &translation);
    }

    if include_current_page_siblings {
        for (source, translation) in page_translated_pairs(scene, current_page, target_ids)? {
            push_unique_context(&mut context, &source, &translation);
        }
    }

    Ok(context)
}

fn page_translated_pairs(
    scene: &Snapshot,
    page: EntityId,
    exclude_contents: &[EntityId],
) -> Result<Vec<(String, String)>> {
    let mut pairs = Vec::new();
    let Some(group) = scene.page(page)?.text_group()? else {
        return Ok(pairs);
    };
    for layer in group.text_layers()? {
        let content = layer.content()?;
        if exclude_contents.contains(&content.id()) {
            continue;
        }
        let Some(source) = content.source()? else {
            continue;
        };
        let Some(translation) = content.translation()? else {
            continue;
        };
        let source_trimmed = source.text.value.trim();
        let translation_trimmed = translation.text.value.trim();
        if !source_trimmed.is_empty() && !translation_trimmed.is_empty() {
            pairs.push((source_trimmed.to_owned(), translation_trimmed.to_owned()));
        }
    }
    Ok(pairs)
}

fn push_unique_context(context: &mut Vec<TranslationContext>, source: &str, translation: &str) {
    if !context
        .iter()
        .any(|existing| existing.source == source && existing.translation == translation)
    {
        context.push(TranslationContext::new(source, translation));
    }
}

#[cfg(test)]
mod tests {
    use koharu_scene::{At, PageDraft, Session, TextLayout, TextLayoutKind};

    use super::*;

    #[tokio::test]
    async fn collects_glossary_preceding_window_and_entity_siblings() {
        let mut session = Session::memory().await.unwrap();
        let mut page_ids = Vec::new();
        let mut target_content = None;
        let patch = session
            .snapshot()
            .patch(|edit| {
                for index in 0..5 {
                    let page =
                        edit.add_page(PageDraft::new(format!("p{index}"), 100.0, 100.0), At::End)?;
                    page_ids.push(page);
                    let content = edit.add_text_content(page, At::End)?;
                    edit.add_text_layer(
                        page,
                        At::End,
                        content,
                        &TextLayout {
                            origin: Origin::User,
                            kind: TextLayoutKind::Paragraph,
                            angle_degrees: None,
                        },
                    )?;
                    edit.set(
                        content,
                        &SourceText {
                            text: Authored::user(format!("src-{index}")),
                            language: Some(LanguageTag::new("ja-JP")?),
                        },
                    )?;
                    if index < 4 {
                        edit.set(
                            content,
                            &Translation {
                                text: Authored::user(format!("dst-{index}")),
                                language: Some(LanguageTag::new("en-US")?),
                            },
                        )?;
                    } else {
                        target_content = Some(content);
                    }
                }
                Ok(())
            })
            .unwrap();
        let snapshot = session.commit(patch).await.unwrap().snapshot;
        let glossary = vec![GlossaryEntry {
            source: "先輩".to_owned(),
            target: "Senpai".to_owned(),
        }];
        let context = collect_translation_context(
            &snapshot,
            page_ids[4],
            &glossary,
            true,
            &[target_content.unwrap()],
        )
        .unwrap();
        assert_eq!(
            context,
            vec![
                TranslationContext::new("先輩", "Senpai"),
                TranslationContext::new("src-1", "dst-1"),
                TranslationContext::new("src-2", "dst-2"),
                TranslationContext::new("src-3", "dst-3"),
            ]
        );
    }
}
