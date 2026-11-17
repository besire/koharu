//! Translation: one conversation per open project, kept in memory, a turn per
//! page in page order. Changing the model, the target, or the prompt starts a
//! new one. A page is its texts as a JSON object numbered in reading order, and
//! its answer must keep the keys, held to a schema; a page that fails never
//! enters the history, and an answer echoing the page before's is asked again
//! without the history it copied. A text given back unchanged, such as a
//! copyright line or a "!?", is in the target's script already, so it keeps
//! its own lettering.

mod conversation;
mod model;
mod placement;
mod turn;

use std::collections::HashMap;

use anyhow::{Context as _, Result, ensure};
use base64::Engine as _;
use icu_experimental::displaynames::{
    DisplayNamesOptions, DisplayNamesPreferences,
    multi::{LanguageDisplayNames, LocaleDisplayNamesFormatter},
};
use icu_locale::LocaleExpander;
use icu_locale_core::{LanguageIdentifier, Locale, langid, subtags::language};
use indoc::formatdoc;
use koharu_llm::rig::{completion::Effort, driver::DynModel};
use koharu_scene::{Content, LayerId, Op, Project};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::{Value, json};
use tokio::sync::OwnedMutexGuard;
use tracing::{Instrument as _, info_span};

pub(crate) use self::conversation::{Conversation, Conversations};
pub use self::placement::placement;
use self::turn::Turn;
use super::{Member, Stopped, Task, action, key};
use crate::{PipelineConfig, Stage};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
pub struct GlossaryEntry {
    pub source: String,
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, specta::Type)]
pub struct InstructionPreset {
    pub name: String,
    pub instructions: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(specta::Type)]
pub struct TranslatorConfig {
    /// A `vendor/model` reference, as koharu-llm connects one.
    pub model: String,
    /// Source language override as a CLDR-named BCP 47 tag; `None` infers it
    /// from the page's recognized blocks and script.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_language"
    )]
    #[specta(type = Option<String>)]
    pub source: Option<LanguageIdentifier>,
    /// A language CLDR names in English.
    #[serde(deserialize_with = "target")]
    #[specta(type = String)]
    pub target: LanguageIdentifier,
    /// The user's own instructions to the model, in any language and form,
    /// such as a name's spelling or the tone; they follow Koharu's own system
    /// prompt as written, and can't replace it.
    pub instructions: String,
    /// Whether each page's turn shows the model the page itself beside its
    /// texts, as an image goes with a chat message to OpenAI's and Gemini's
    /// models; only for a model that reads images.
    pub vision: bool,
    /// How much the model reasons before it answers, by rig's portable levels
    /// after OpenAI's `reasoning_effort`; the model's own default when unset.
    #[specta(type = Option<String>)]
    pub reasoning: Option<Effort>,
    /// Recurring terms and character names appended to the system prompt.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub glossary: Vec<GlossaryEntry>,
    /// Named instruction presets saved in preferences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub presets: Vec<InstructionPreset>,
}

impl Default for TranslatorConfig {
    fn default() -> Self {
        Self {
            // On 16 Blue Archive pages 9B read names and meaning that 2B
            // missed, such as Shale for シャーレ where 2B wrote "tray", at
            // about twice the decoding time.
            model: "koharu/index-translate-9b".to_owned(),
            source: None,
            target: langid!("en"),
            instructions: String::new(),
            vision: false,
            reasoning: None,
            glossary: Vec::new(),
            presets: Vec::new(),
        }
    }
}

/// The system prompt: Index-Translate's structured-data template in English,
/// as bilibili/Index-Translate's `docs/prompts.md` gives it
/// (93ac63774d1c6939aec05e6f0f191bf90f225365, Apache-2.0), for the JSON the
/// answer is parsed by; then the resolved source language (if known), user's
/// instructions as written, and any glossary entries.
fn system_prompt(config: &TranslatorConfig, source: Option<&LanguageIdentifier>) -> Result<String> {
    let mut system = formatdoc! {"
        Translate the manga text in each message into {target}. Each message is a page's texts as a JSON object, numbered in reading order: translate only the values, and never alter the structure or the keys. Output only the translated JSON, without any explanation.",
        target = name(&config.target)?,
    };
    if let Some(source) = source {
        system.push_str(&format!("\n\nSource language: {}.", name(source)?));
    }
    let instructions = config.instructions.trim();
    if !instructions.is_empty() {
        system.push_str(&format!("\n\n{instructions}"));
    }
    let entries: Vec<String> = config
        .glossary
        .iter()
        .filter_map(|entry| {
            let source = entry.source.trim();
            let target = entry.target.trim();
            if source.is_empty() || target.is_empty() {
                return None;
            }
            match entry
                .note
                .as_deref()
                .map(str::trim)
                .filter(|note| !note.is_empty())
            {
                Some(note) => Some(format!("- {source} -> {target} ({note})")),
                None => Some(format!("- {source} -> {target}")),
            }
        })
        .collect();
    if !entries.is_empty() {
        system.push_str(&format!(
            "\n\n[Glossary]\nUse these exact translations when the source term appears:\n{}",
            entries.join("\n")
        ));
    }
    Ok(system)
}

/// Resolves a page's source language: explicit `configured` wins, then
/// unambiguous Kana/Hangul script across `texts`, then the dominant determined
/// block tag, and finally Han/Latin script fallback when OCR left `und`.
fn resolve_source_language(
    configured: Option<&LanguageIdentifier>,
    blocks: &[(LayerId, LanguageIdentifier)],
    texts: &[String],
) -> Option<LanguageIdentifier> {
    if let Some(configured) = configured {
        return Some(configured.clone());
    }
    let (mut kana, mut hangul, mut han, mut latin) = (0usize, 0usize, 0usize, 0usize);
    for ch in texts.iter().flat_map(|text| text.chars()) {
        if matches!(
            ch,
            '\u{3040}'..='\u{309F}'
                | '\u{30A0}'..='\u{30FF}'
                | '\u{31F0}'..='\u{31FF}'
                | '\u{FF66}'..='\u{FF9F}'
        ) {
            kana += 1;
        } else if matches!(
            ch,
            '\u{AC00}'..='\u{D7AF}' | '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}'
        ) {
            hangul += 1;
        } else if matches!(ch, '\u{3400}'..='\u{4DBF}' | '\u{4E00}'..='\u{9FFF}') {
            han += 1;
        } else if ch.is_ascii_alphabetic() {
            latin += 1;
        }
    }
    if kana > 0 && kana >= hangul {
        return Some(langid!("ja"));
    }
    if hangul > 0 {
        return Some(langid!("ko"));
    }
    let mut counts: HashMap<&LanguageIdentifier, usize> = HashMap::new();
    for (_, language) in blocks {
        if *language != LanguageIdentifier::UNKNOWN {
            *counts.entry(language).or_default() += 1;
        }
    }
    if let Some((&dominant, _)) = counts.iter().max_by_key(|(_, count)| *count) {
        return Some(dominant.clone());
    }
    if han > 0 {
        Some(langid!("zh"))
    } else if latin > 0 {
        Some(langid!("en"))
    } else {
        None
    }
}

fn target<'de, D: Deserializer<'de>>(deserializer: D) -> Result<LanguageIdentifier, D::Error> {
    let target = LanguageIdentifier::deserialize(deserializer)?;
    name(&target).map_err(D::Error::custom)?;
    Ok(target)
}

fn optional_language<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<LanguageIdentifier>, D::Error> {
    let Some(language) = Option::<LanguageIdentifier>::deserialize(deserializer)? else {
        return Ok(None);
    };
    name(&language).map_err(D::Error::custom)?;
    Ok(Some(language))
}

/// A page's texts, notes, and the page as a JPEG in base 64 when the model sees it.
struct Page {
    project: Project,
    /// Each block and its source's language.
    blocks: Vec<(LayerId, LanguageIdentifier)>,
    texts: Vec<String>,
    notes: Vec<Option<String>>,
    image: Option<String>,
}

pub(crate) async fn translate(
    task: &Task,
    config: &PipelineConfig,
    mut conversation: OwnedMutexGuard<Conversation>,
) -> Result<()> {
    let member = &task.members[0];
    let translator = config.translator.clone();
    let (read, vision) = (member.clone(), translator.vision);
    let page = tokio::task::spawn_blocking(move || {
        let _span = info_span!("read", stage = %Stage::Translate, page = read.index + 1).entered();
        page(&read, vision)
    })
    .await??;
    let Some(page) = page else {
        return Ok(());
    };
    let conversation = &mut *conversation;
    conversation.keep(&translator);
    conversation.hydrate(&page.project, member.page);
    let model = match key(Stage::Translate, config) {
        // A local model loads on the GPU thread, which only hands it over:
        // its engine runs on a thread of its own, so the turn waits off the
        // GPU thread, and the other stages' models run meanwhile.
        Some(key) => {
            let (reference, images) = (translator.model.clone(), translator.vision);
            task.inference(&[member], move |models| {
                let model = models.get(&key, |_| model::load(&reference, images))?;
                Ok(DynModel::clone(model))
            })
            .await?
        }
        None => {
            task.started(&[member]);
            conversation.hosted(&translator).await?
        }
    };
    let resolved_source =
        resolve_source_language(translator.source.as_ref(), &page.blocks, &page.texts);
    let system = system_prompt(&translator, resolved_source.as_ref())?;
    let turn = Turn::new(
        system,
        &page.texts,
        &page.notes,
        page.image.as_deref(),
        translator.reasoning,
        &member.cancel,
    );
    let Some(translations) = turn
        .take(conversation, &model)
        .instrument(info_span!("llm", texts = page.texts.len()))
        .await?
    else {
        return Err(Stopped.into());
    };
    let translated: Vec<(LayerId, LanguageIdentifier, String)> = page
        .blocks
        .into_iter()
        .zip(translations)
        .map(|((block, source), text)| {
            let source = translator
                .source
                .clone()
                .or_else(|| (source != LanguageIdentifier::UNKNOWN).then_some(source))
                .or_else(|| resolved_source.clone())
                .unwrap_or(LanguageIdentifier::UNKNOWN);
            (block, source, text)
        })
        .collect();
    member
        .apply(Stage::Translate, move |project| {
            written(project, translated, &translator)
        })
        .await
}

/// The page's texts the task works on, and the page itself when the model
/// sees it; `None` when it has no text.
fn page(member: &Member, vision: bool) -> Result<Option<Page>> {
    // A page with no scan, such as a blank one with typed text, goes
    // without its image.
    let seen = if vision { member.scan()? } else { None };
    let (project, image) = match seen {
        Some((project, scan)) => (project, Some(sight(&scan)?)),
        None => (member.session.lock().project(), None),
    };
    let mut blocks = Vec::new();
    let mut texts = Vec::new();
    let mut notes = Vec::new();
    for (id, block) in member.blocks(&project) {
        let text = block.source.content.text();
        if text.trim().is_empty() {
            continue;
        }
        let note = project
            .layer(id)
            .ok()
            .and_then(|layer| layer.attributes.get("koharu"))
            .and_then(|namespace| namespace.get("note"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|note| !note.is_empty())
            .map(str::to_owned);
        blocks.push((id, block.source.language.clone()));
        texts.push(text);
        notes.push(note);
    }
    Ok((!texts.is_empty()).then_some(Page {
        project,
        blocks,
        texts,
        notes,
        image,
    }))
}

/// The longest side of the page a model sees, in pixels: about 1,120 image
/// tokens to Index-Translate, above the 1,024 llama.cpp asks of Qwen-VL
/// models, where the whole scan took 4,050 and read no better.
const SIGHT: u32 = 1280;

/// `scan` at `SIGHT` as a JPEG in base 64. Each pixel averages the scan's
/// under it: a fraction of Lanczos's time, and the page is on the
/// translation's critical path.
fn sight(scan: &image::DynamicImage) -> Result<String> {
    let page = scan.thumbnail(SIGHT, SIGHT).to_rgb8();
    let mut bytes = Vec::new();
    page.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Jpeg,
    )?;
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

/// The ops writing each block's translation, placed, into the project as it
/// is now.
fn written(
    project: &Project,
    translated: Vec<(LayerId, LanguageIdentifier, String)>,
    config: &TranslatorConfig,
) -> Result<Vec<Op>> {
    let target = &config.target;
    // Each block's placement reads the project as it is before any of them,
    // so they're found in parallel: a balloon's room split is a flood over
    // its pixels, run for each of its texts.
    let placed: Vec<_> = translated
        .into_par_iter()
        .filter_map(|(block, source, text)| {
            let Ok(Content::Block(found)) = project.layer(block).map(|layer| &layer.content) else {
                return None;
            };
            // Its clean-up would erase the lettering for the translation to
            // redraw it in another hand.
            let unchanged = text.trim() == found.source.content.text().trim();
            let placed = (!unchanged).then(|| placement(project, block, target));
            Some((block, source, text, placed))
        })
        .collect();
    let mut ops = Vec::new();
    for (block, source, text, placed) in placed {
        let Some(placed) = placed else {
            ops.push(Op::UpdateLayer {
                layer: block.into(),
                patch: json!({"translation": null, "cleanup": {"visible": false}}),
            });
            continue;
        };
        ops.push(Op::SetText {
            layer: block.into(),
            text,
        });
        ops.push(placed?);
        ops.push(Op::UpdateLayer {
            layer: block.into(),
            patch: json!({"translation": {"language": target}, "cleanup": {"visible": true}}),
        });
        let parameters = json!({"sourceLanguage": source, "targetLanguage": target});
        ops.push(Op::SetAttribute {
            layer: block.into(),
            namespace: "c2pa".to_owned(),
            key: "translation".to_owned(),
            value: Some(action("c2pa.translated", &config.model, Some(parameters))),
        });
    }
    Ok(ops)
}

/// The target's plain English name from CLDR, such as Simplified Chinese.
/// Chinese is named by its script, which its region implies, and another
/// language drops a region its language implies, as ja-JP is Japanese. A
/// language CLDR doesn't name fails, rather than going in as its tag.
fn name(target: &LanguageIdentifier) -> Result<String> {
    let mut target = target.clone();
    let expander = LocaleExpander::new_extended();
    if target.language == language!("zh") {
        expander.maximize(&mut target);
        target.region = None;
    } else {
        expander.minimize(&mut target);
    }
    let english = DisplayNamesPreferences::from(&langid!("en"));
    let options = DisplayNamesOptions::default();
    let languages = LanguageDisplayNames::try_new(english, options)
        .context("couldn't load CLDR's language names")?;
    ensure!(
        languages.of(target.language).is_some(),
        "CLDR has no English name for {target}"
    );
    let locales = LocaleDisplayNamesFormatter::try_new(english, options)
        .context("couldn't load CLDR's locale names")?;
    Ok(locales.of(&Locale::from(target)).into_owned())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use koharu_scene::{Created, Edit, Session};

    use super::*;

    #[test]
    fn a_language_cldr_doesnt_name_fails() {
        let language = LanguageIdentifier::try_from_str("xx").expect("a tag");
        assert!(name(&language).is_err());
    }

    #[test]
    fn instructions_follow_the_whole_template() {
        let template = system_prompt(&TranslatorConfig::default(), None).expect("English names it");
        let config = TranslatorConfig {
            instructions: indoc! {"

                シャーレはSchaleと訳す。

                敬称は残す。

            "}
            .to_owned(),
            ..Default::default()
        };
        assert_eq!(
            system_prompt(&config, None).expect("English names it"),
            formatdoc! {"
                {template}

                シャーレはSchaleと訳す。

                敬称は残す。",
                template = template,
            }
        );
    }

    #[test]
    fn source_language_and_glossary_extend_the_system_prompt() -> Result<()> {
        let blocks = Vec::new();
        assert_eq!(
            resolve_source_language(None, &blocks, &["잠깐만요!".to_owned()]),
            Some(langid!("ko"))
        );
        assert_eq!(
            resolve_source_language(None, &blocks, &["你好世界".to_owned()]),
            Some(langid!("zh"))
        );
        assert_eq!(
            resolve_source_language(Some(&langid!("en")), &blocks, &["こんにちは".to_owned()]),
            Some(langid!("en"))
        );

        let config = TranslatorConfig {
            source: Some(langid!("ja")),
            target: langid!("zh-Hans"),
            glossary: vec![GlossaryEntry {
                source: "冬優子".to_owned(),
                target: "冬优子".to_owned(),
                note: Some("female idol".to_owned()),
            }],
            ..TranslatorConfig::default()
        };
        let prompt = system_prompt(&config, config.source.as_ref())?;
        assert!(prompt.contains("Simplified Chinese"));
        assert!(prompt.contains("Source language: Japanese."));
        assert!(prompt.contains("- 冬優子 -> 冬优子 (female idol)"));
        Ok(())
    }

    #[test]
    fn reopening_a_project_hydrates_prior_translated_pages_into_conversation() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let mut session = Session::create(dir.path().join("test.khr"))?;
        let change = session.apply(Edit {
            label: "seed".to_owned(),
            ops: vec![
                serde_json::from_value(
                    json!({"op": "add_page", "id": "$p1", "index": 0, "size": {"width": 200, "height": 200}}),
                )?,
                serde_json::from_value(
                    json!({"op": "add_page", "id": "$p2", "index": 1, "size": {"width": 200, "height": 200}}),
                )?,
                serde_json::from_value(json!({
                    "op": "add_layer",
                    "page": "$p1",
                    "index": 0,
                    "layer": {
                        "kind": "block",
                        "source": {"content": "冬優子だよ"},
                        "translation": {"content": "我是冬优子"},
                    }
                }))?,
            ],
        })?;
        let Some(Created::Page { id: second, .. }) = change.created.get(1) else {
            anyhow::bail!("missing second page");
        };
        let mut conversations = Conversations::default();
        let mut guard = conversations.lock(session.project().id()).context("lock")?;
        guard.keep(&TranslatorConfig::default());
        guard.hydrate(&session.project(), *second);
        assert_eq!(guard.context.messages().len(), 2);
        Ok(())
    }
}
