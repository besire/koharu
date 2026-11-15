//! Translation through local and hosted providers.

mod backend;
mod error;
mod language;
mod local;
mod model;
mod prompt;
mod provider;
mod remote;

use std::{sync::Arc, time::Duration};

use koharu_ml::Device;

use error::{Error, Result};
use local::LocalTranslator;

pub use backend::{TranslationContext, TranslationRequest};
pub use language::Language;
pub use model::{GenerationConfig, Model, ModelSelection, Quantization};
pub(crate) use model::{ModelGeneration, QuantizationDefinition, display_name};
pub use provider::{Provider, ProviderConfig, ProvidersConfig};

#[derive(Clone)]
pub struct Translator {
    providers: koharu_config::Config<ProvidersConfig>,
    local: Arc<tokio::sync::Mutex<Option<LoadedLocal>>>,
    client: reqwest::Client,
    device: Device,
}

struct LoadedLocal {
    model: Option<String>,
    quantization: Option<String>,
    translator: Arc<LocalTranslator>,
}

impl LoadedLocal {
    fn matches(&self, selection: &ModelSelection) -> bool {
        self.model == selection.model && self.quantization == selection.quantization
    }
}

impl Translator {
    pub fn from_config(
        device: Device,
        providers: koharu_config::Config<ProvidersConfig>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            providers,
            local: Arc::new(tokio::sync::Mutex::new(None)),
            client: koharu_runtime::http_client()?,
            device,
        })
    }

    #[must_use]
    pub fn model(selection: &ModelSelection) -> &'static str {
        selection.provider.into()
    }

    #[must_use]
    pub fn supports_vision(selection: &ModelSelection, generation: &GenerationConfig) -> bool {
        generation.vision.unwrap_or(false)
            && (selection.provider != Provider::Local || local::supports_vision(selection))
    }

    #[must_use]
    pub fn loaded(&self, selection: &ModelSelection) -> bool {
        if selection.provider != Provider::Local {
            return true;
        }
        self.local
            .try_lock()
            .map(|loaded| {
                loaded
                    .as_ref()
                    .is_some_and(|loaded| loaded.matches(selection))
            })
            .unwrap_or(true)
    }

    pub fn unload(&self) -> bool {
        self.local
            .try_lock()
            .map(|mut loaded| loaded.take().is_some())
            .unwrap_or(false)
    }

    #[tracing::instrument(skip_all)]
    pub async fn load_model(&self, selection: &ModelSelection) -> anyhow::Result<()> {
        if selection.provider == Provider::Local {
            self.local(selection).await?;
        }
        Ok(())
    }

    pub async fn translate(
        &self,
        selection: &ModelSelection,
        generation: GenerationConfig,
        mut request: TranslationRequest,
    ) -> anyhow::Result<(&'static str, Vec<String>)> {
        let provider = selection.provider;
        let provider_id: &'static str = provider.into();
        if request.segments.is_empty() {
            return Ok((provider_id, request.segments));
        }

        let generation = generation.for_model(selection);

        if Self::supports_vision(selection, &generation) {
            request.prepare_image()?;
        } else {
            request.remove_image();
        }

        let expected = request.segments.len();
        let mut parsed = self
            .translate_once(selection, &generation, &request)
            .await?;
        if parsed.segments.len() != expected {
            return Err(Error::SegmentCount {
                provider: provider_id,
                expected,
                actual: parsed.segments.len(),
            }
            .into());
        }

        if supports_missing_segment_retry(provider) && !parsed.missing_ids.is_empty() {
            let retry_request = build_missing_retry_request(&request, &parsed);
            let retry_parsed = self
                .translate_once(selection, &generation, &retry_request)
                .await?;
            if retry_parsed.segments.len() != retry_request.segments.len() {
                return Err(Error::SegmentCount {
                    provider: provider_id,
                    expected: retry_request.segments.len(),
                    actual: retry_parsed.segments.len(),
                }
                .into());
            }
            apply_missing_retry(&mut parsed, &retry_parsed);
        }

        Ok((provider_id, parsed.segments))
    }

    async fn translate_once(
        &self,
        selection: &ModelSelection,
        generation: &GenerationConfig,
        request: &TranslationRequest,
    ) -> Result<prompt::ParsedTranslations> {
        if selection.provider == Provider::Local {
            self.local(selection)
                .await?
                .translate(request.clone(), *generation)
                .await
        } else {
            let providers = self.providers.read()?.clone();
            remote::translate(&self.client, &providers, selection, generation, request).await
        }
    }

    #[tracing::instrument(skip_all)]
    pub async fn models() -> anyhow::Result<Vec<Model>> {
        static CLIENT: tokio::sync::OnceCell<reqwest::Client> = tokio::sync::OnceCell::const_new();

        let providers = ProvidersConfig::load()?;
        let providers = providers.read()?.clone();
        let client = CLIENT
            .get_or_try_init(|| async {
                reqwest::Client::builder()
                    .timeout(Duration::from_secs(5))
                    .build()
            })
            .await?;
        let mut models = local::models();
        models.extend(remote::models(client, &providers).await);
        Ok(models)
    }

    async fn local(&self, selection: &ModelSelection) -> Result<Arc<LocalTranslator>> {
        let mut loaded = self.local.lock().await;
        if loaded
            .as_ref()
            .is_none_or(|loaded| !loaded.matches(selection))
        {
            *loaded = Some(LoadedLocal {
                model: selection.model.clone(),
                quantization: selection.quantization.clone(),
                translator: Arc::new(LocalTranslator::load(self.device.clone(), selection).await?),
            });
        }
        Ok(Arc::clone(
            &loaded
                .as_ref()
                .expect("local translator was loaded")
                .translator,
        ))
    }
}

fn supports_missing_segment_retry(provider: Provider) -> bool {
    !matches!(
        provider,
        Provider::DeepL | Provider::GoogleCloudTranslation | Provider::Caiyun
    )
}

fn build_missing_retry_request(
    request: &TranslationRequest,
    parsed: &prompt::ParsedTranslations,
) -> TranslationRequest {
    let mut context = request.context.clone();
    for (id, (source, translated)) in request.segments.iter().zip(&parsed.segments).enumerate() {
        if !parsed.missing_ids.contains(&id) {
            context.push(TranslationContext::new(source.clone(), translated.clone()));
        }
    }

    let segments = parsed
        .missing_ids
        .iter()
        .map(|&id| request.segments[id].clone());
    let notes = parsed
        .missing_ids
        .iter()
        .map(|&id| request.segment_notes.get(id).cloned().flatten());

    let mut retry = TranslationRequest::new(segments, request.target_language)
        .with_context(context)
        .with_segment_notes(notes);
    if let Some(source_language) = request.source_language {
        retry = retry.with_source_language(source_language);
    }
    if let Some(instructions) = &request.instructions {
        retry = retry.with_instructions(instructions.clone());
    }
    retry
}

fn apply_missing_retry(
    parsed: &mut prompt::ParsedTranslations,
    retry_parsed: &prompt::ParsedTranslations,
) {
    let original_missing = parsed.missing_ids.clone();
    let mut remaining_missing = Vec::new();
    for (retry_index, original_id) in original_missing.into_iter().enumerate() {
        if retry_parsed.missing_ids.contains(&retry_index) {
            remaining_missing.push(original_id);
        } else if let Some(value) = retry_parsed.segments.get(retry_index) {
            parsed.segments[original_id] = value.clone();
        }
    }
    parsed.missing_ids = remaining_missing;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local_selection(model: &str) -> ModelSelection {
        ModelSelection {
            provider: Provider::Local,
            model: Some(model.to_owned()),
            quantization: None,
            vision: true,
            reasoning: true,
        }
    }

    #[test]
    fn local_vision_requires_capability_and_generation_setting() {
        assert!(Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("gemma4-e2b-it"),
            &GenerationConfig {
                vision: Some(false),
                ..GenerationConfig::default()
            }
        ));
        assert!(!Translator::supports_vision(
            &local_selection("lfm2.5-1.2b-instruct"),
            &GenerationConfig {
                vision: Some(true),
                ..GenerationConfig::default()
            }
        ));
    }

    #[test]
    fn missing_segment_retry_builds_focused_request_and_splices_results() {
        assert!(supports_missing_segment_retry(Provider::DeepSeek));
        assert!(!supports_missing_segment_retry(Provider::DeepL));

        let request = TranslationRequest::new(["a", "b", "c"], Language::English)
            .with_source_language(Language::Japanese)
            .with_context([TranslationContext::new("prev", "prior")])
            .with_segment_notes([None, Some("Shout".to_owned()), None])
            .with_image(Arc::new(image::DynamicImage::new_rgb8(2, 2)));
        let mut parsed = prompt::ParsedTranslations {
            segments: vec!["A".to_owned(), "b".to_owned(), "C".to_owned()],
            missing_ids: vec![1],
        };

        let retry = build_missing_retry_request(&request, &parsed);
        assert_eq!(retry.segments, vec!["b".to_owned()]);
        assert_eq!(retry.segment_notes, vec![Some("Shout".to_owned())]);
        assert!(retry.image.is_none());
        assert_eq!(retry.context.len(), 3);

        let retry_parsed = prompt::ParsedTranslations {
            segments: vec!["B!".to_owned()],
            missing_ids: Vec::new(),
        };
        apply_missing_retry(&mut parsed, &retry_parsed);
        assert_eq!(
            parsed.segments,
            vec!["A".to_owned(), "B!".to_owned(), "C".to_owned()]
        );
        assert!(parsed.missing_ids.is_empty());
    }
}
