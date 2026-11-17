'use client'

import { ChevronDown, Plus, Trash2 } from 'lucide-react'
import { type ReactNode, useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ModelPicker } from '@/components/controls/ModelPicker'
import { detectors, inpainters, recognizers, vendorName } from '@/components/preferences/models'
import {
  NumberField,
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { modelName, orderedLanguageChoices, REASONING, vendorOf } from '@/lib/translation'
import type {
  Detector,
  GlossaryEntry,
  Inpainter,
  Model,
  PipelineConfig,
  Recognizer,
  TranslatorConfig,
} from '@koharu/ipc/protocol'
import { Badge } from '@koharu/ui/components/badge'
import { Button } from '@koharu/ui/components/button'
import { Input } from '@koharu/ui/components/input'
import { Popover, PopoverContent, PopoverTrigger } from '@koharu/ui/components/popover'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'
import { Switch } from '@koharu/ui/components/switch'
import { Textarea } from '@koharu/ui/components/textarea'

type StageName = 'detect' | 'recognize' | 'inpaint' | 'translate'

/** The `[pipeline]` stages in the order they run, a section each with its model first: the
 * detector with its thresholds, the recognizer, the inpainter with its prompt, and the translator
 * with its options. */
export function PipelinePreferences({
  value,
  models,
  onChange,
}: {
  value: PipelineConfig
  models: Model[]
  onChange: (value: PipelineConfig) => void
}) {
  const { t } = useTranslation()
  const detector = value.detector ?? {}
  const inpainter = value.inpainter ?? {}
  const translator = value.translator ?? {}
  const thresholds = ['text', 'onomatopoeia', 'bubble', 'panel'] as const

  return (
    <PreferencePage
      title={t('settings.pipeline.title')}
      description={t('settings.pipeline.description')}
    >
      <Stage
        stage='detect'
        model={
          <ModelSelect
            stage='detect'
            value={detector.model ?? 'koharu-layout-highres'}
            names={detectors}
            onChange={(model) => onChange({ ...value, detector: { ...detector, model } })}
          />
        }
      >
        <PreferenceRow title={t('settings.pipeline.thresholds')} align='start'>
          <div className='grid grid-cols-2 gap-2'>
            {thresholds.map((threshold) => (
              <NumberField
                key={threshold}
                label={t(`settings.pipeline.options.${threshold}Threshold`)}
                value={detector[threshold] ?? null}
                min={0}
                max={1}
                step={0.05}
                onChange={(next) =>
                  onChange({ ...value, detector: { ...detector, [threshold]: next } })
                }
              />
            ))}
          </div>
        </PreferenceRow>
      </Stage>
      <Stage
        stage='recognize'
        model={
          <ModelSelect
            stage='recognize'
            value={value.recognizer?.model ?? 'manga-ocr-ar'}
            names={recognizers}
            onChange={(model: Recognizer) => onChange({ ...value, recognizer: { model } })}
          />
        }
      />
      <Stage
        stage='inpaint'
        model={
          <ModelSelect
            stage='inpaint'
            value={inpainter.model ?? 'lama'}
            names={inpainters}
            onChange={(model: Inpainter) =>
              onChange({ ...value, inpainter: { ...inpainter, model } })
            }
          />
        }
      >
        <PreferenceRow
          title={t('settings.pipeline.options.prompt')}
          description={t('settings.pipeline.options.promptDescription')}
          align='start'
        >
          <Textarea
            aria-label={t('settings.pipeline.options.prompt')}
            value={inpainter.prompt ?? ''}
            className='field-sizing-fixed min-h-28 resize-y overflow-y-auto text-[12px] leading-5'
            onChange={(event) =>
              onChange({
                ...value,
                inpainter: { ...inpainter, prompt: event.currentTarget.value },
              })
            }
          />
        </PreferenceRow>
      </Stage>
      <Stage
        stage='translate'
        model={
          <TranslatorModel
            value={translator.model ?? ''}
            models={models}
            label={t('settings.pipeline.modelLabel', {
              stage: t('settings.pipeline.stages.translate.title'),
            })}
            onChange={(model) => onChange({ ...value, translator: { ...translator, model } })}
          />
        }
      >
        <TranslatorOptions
          value={translator}
          onChange={(next) => onChange({ ...value, translator: next })}
        />
      </Stage>
    </PreferencePage>
  )
}

/** A stage's section: what it does, its model, then its options. */
function Stage({
  stage,
  model,
  children,
}: {
  stage: StageName
  model: ReactNode
  children?: ReactNode
}) {
  const { t } = useTranslation()
  return (
    <PreferenceSection
      title={t(`settings.pipeline.stages.${stage}.title`)}
      description={t(`settings.pipeline.stages.${stage}.description`)}
    >
      <PreferenceRow title={t('settings.pipeline.model')}>{model}</PreferenceRow>
      {children}
    </PreferenceSection>
  )
}

function ModelSelect<Model extends Detector | Recognizer | Inpainter>({
  stage,
  value,
  names,
  onChange,
}: {
  stage: Exclude<StageName, 'translate'>
  value: Model
  names: Record<Model, string>
  onChange: (model: Model) => void
}) {
  const { t } = useTranslation()
  const models = Object.keys(names) as Model[]
  return (
    <Select
      value={value}
      items={names}
      onValueChange={(model) => model && onChange(model as Model)}
    >
      <SelectTrigger
        aria-label={t('settings.pipeline.modelLabel', {
          stage: t(`settings.pipeline.stages.${stage}.title`),
        })}
        className='h-8 w-full text-[11px]'
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {models.map((model) => (
          <SelectItem key={model} value={model}>
            {names[model]}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** The translator's model, any provider's, labelled with where it runs. */
function TranslatorModel({
  value,
  models,
  label,
  onChange,
}: {
  value: string
  models: Model[]
  label: string
  onChange: (model: string) => void
}) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        type='button'
        aria-label={label}
        className='flex h-8 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-transparent px-2.5 text-[11px] transition-colors outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50'
      >
        <span className='flex min-w-0 flex-1 items-center gap-2 text-left'>
          {value && (
            <Badge variant='outline' className='shrink-0 px-1.5 py-0 text-[9px] font-medium'>
              {vendorName(vendorOf(value), t)}
            </Badge>
          )}
          <span className='truncate'>{modelName(models, value) || t('inference.noModel')}</span>
        </span>
        <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
      </PopoverTrigger>
      <PopoverContent
        align='start'
        sideOffset={4}
        className='w-(--anchor-width) min-w-64 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
      >
        <ModelPicker
          value={value}
          models={models}
          onBack={() => setOpen(false)}
          onSelect={(reference) => {
            onChange(reference)
            setOpen(false)
          }}
        />
      </PopoverContent>
    </Popover>
  )
}

/** The translator's options under `[pipeline.translator]`, as rows of its section: whether the
 * model sees each page, how much it reasons, the target language, and the user's instructions,
 * which Koharu adds to its own fixed system prompt. */
function TranslatorOptions({
  value,
  onChange,
}: {
  value: TranslatorConfig
  onChange: (value: TranslatorConfig) => void
}) {
  const { t, i18n } = useTranslation()
  const [presetName, setPresetName] = useState('')
  const languageChoices = useMemo(() => orderedLanguageChoices(i18n.language), [i18n.language])
  const sourceItems = useMemo(
    () =>
      Object.fromEntries([
        ['auto', t('model.autoDetect')],
        ...languageChoices.map((language) => [language.tag, language.name]),
      ]),
    [languageChoices, t],
  )
  const reasoning = value.reasoning ?? 'default'
  const presets = value.presets ?? []
  const glossary = value.glossary ?? []
  const activePreset =
    presets.find((preset) => preset.instructions === (value.instructions ?? ''))?.name ?? ''

  const updateGlossary = (index: number, patch: Partial<GlossaryEntry>) => {
    onChange({
      ...value,
      glossary: glossary.map((entry, candidate) =>
        candidate === index ? { ...entry, ...patch } : entry,
      ),
    })
  }

  return (
    <>
      <PreferenceRow
        title={t('settings.pipeline.translator.vision')}
        description={t('settings.pipeline.translator.visionDescription')}
      >
        <Switch
          aria-label={t('settings.pipeline.translator.vision')}
          checked={value.vision ?? false}
          onCheckedChange={(vision) => onChange({ ...value, vision })}
        />
      </PreferenceRow>
      <PreferenceRow
        title={t('settings.pipeline.translator.reasoning')}
        description={t('settings.pipeline.translator.reasoningDescription')}
      >
        <Select
          value={reasoning}
          items={Object.fromEntries([
            ['default', t('inference.providerDefault')],
            ...REASONING.map((level) => [level, t(`reasoning.${level}`)]),
          ])}
          onValueChange={(level) =>
            level && onChange({ ...value, reasoning: level === 'default' ? null : level })
          }
        >
          <SelectTrigger
            aria-label={t('settings.pipeline.translator.reasoning')}
            className='h-8 w-full text-[11px]'
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value='default'>{t('inference.providerDefault')}</SelectItem>
            {REASONING.map((level) => (
              <SelectItem key={level} value={level}>
                {t(`reasoning.${level}`)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </PreferenceRow>
      <PreferenceRow
        title={t('model.sourceLanguage')}
        description={t('settings.pipeline.translator.sourceLanguageDescription')}
      >
        <Select
          value={value.source ?? 'auto'}
          items={sourceItems}
          onValueChange={(source) =>
            source && onChange({ ...value, source: source === 'auto' ? null : source })
          }
        >
          <SelectTrigger aria-label={t('model.sourceLanguage')} className='h-8 w-full text-[11px]'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value='auto'>{t('model.autoDetect')}</SelectItem>
            {languageChoices.map((language) => (
              <SelectItem key={language.tag} value={language.tag}>
                {language.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </PreferenceRow>
      <PreferenceRow
        title={t('model.targetLanguage')}
        description={t('settings.pipeline.translator.targetLanguageDescription')}
      >
        <Select
          value={value.target ?? 'en'}
          items={Object.fromEntries(
            languageChoices.map((language) => [language.tag, language.name]),
          )}
          onValueChange={(target) => target && onChange({ ...value, target })}
        >
          <SelectTrigger aria-label={t('model.targetLanguage')} className='h-8 w-full text-[11px]'>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {languageChoices.map((language) => (
              <SelectItem key={language.tag} value={language.tag}>
                {language.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </PreferenceRow>
      <PreferenceRow
        title={t('settings.pipeline.translator.instructions')}
        description={t('settings.pipeline.translator.instructionsDescription')}
        align='start'
      >
        <div className='grid gap-1.5'>
          {presets.length > 0 && (
            <div className='flex items-center gap-1.5'>
              <Select
                value={activePreset || '__custom__'}
                items={Object.fromEntries([
                  ['__custom__', t('settings.pipeline.translator.presets')],
                  ...presets.map((preset) => [preset.name, preset.name]),
                ])}
                onValueChange={(selected) => {
                  const match = presets.find((preset) => preset.name === selected)
                  if (match) onChange({ ...value, instructions: match.instructions })
                }}
              >
                <SelectTrigger
                  aria-label={t('settings.pipeline.translator.presets')}
                  className='h-7 flex-1 text-[11px]'
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {presets.map((preset) => (
                    <SelectItem key={preset.name} value={preset.name}>
                      {preset.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                type='button'
                variant='ghost'
                size='icon-sm'
                disabled={!activePreset}
                aria-label={t('settings.pipeline.translator.deletePreset')}
                title={t('settings.pipeline.translator.deletePreset')}
                className='size-7 shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive'
                onClick={() =>
                  onChange({
                    ...value,
                    presets: presets.filter((preset) => preset.name !== activePreset),
                  })
                }
              >
                <Trash2 className='size-3.5' />
              </Button>
            </div>
          )}
          <Textarea
            aria-label={t('settings.pipeline.translator.instructions')}
            value={value.instructions ?? ''}
            className='field-sizing-fixed min-h-28 resize-y overflow-y-auto text-[12px] leading-5'
            onChange={(event) => onChange({ ...value, instructions: event.currentTarget.value })}
          />
          <div className='flex items-center gap-1.5'>
            <Input
              value={presetName}
              aria-label={t('settings.pipeline.translator.presetName')}
              placeholder={t('settings.pipeline.translator.presetName')}
              className='h-7 flex-1 text-[11px]'
              onChange={(event) => setPresetName(event.currentTarget.value)}
            />
            <Button
              type='button'
              variant='outline'
              size='xs'
              disabled={!presetName.trim() || !(value.instructions ?? '').trim()}
              className='h-7 shrink-0 px-2.5 text-[11px]'
              onClick={() => {
                const name = presetName.trim()
                const instructions = value.instructions ?? ''
                if (!name || !instructions.trim()) return
                const nextPresets = presets.some((preset) => preset.name === name)
                  ? presets.map((preset) =>
                      preset.name === name ? { name, instructions } : preset,
                    )
                  : [...presets, { name, instructions }]
                onChange({ ...value, presets: nextPresets })
                setPresetName('')
              }}
            >
              {t('settings.pipeline.translator.savePreset')}
            </Button>
          </div>
        </div>
      </PreferenceRow>
      <PreferenceRow
        title={t('settings.pipeline.translator.glossary')}
        description={t('settings.pipeline.translator.glossaryDescription')}
        align='start'
      >
        <div className='grid gap-2'>
          {glossary.map((entry, index) => (
            <div
              key={index}
              className='grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_auto] items-center gap-1.5'
            >
              <Input
                value={entry.source}
                aria-label={t('settings.pipeline.translator.glossarySource')}
                placeholder={t('settings.pipeline.translator.glossarySource')}
                className='h-8 text-[11px]'
                onChange={(event) => updateGlossary(index, { source: event.currentTarget.value })}
              />
              <Input
                value={entry.target}
                aria-label={t('settings.pipeline.translator.glossaryTarget')}
                placeholder={t('settings.pipeline.translator.glossaryTarget')}
                className='h-8 text-[11px]'
                onChange={(event) => updateGlossary(index, { target: event.currentTarget.value })}
              />
              <Input
                value={entry.note ?? ''}
                aria-label={t('settings.pipeline.translator.glossaryNote')}
                placeholder={t('settings.pipeline.translator.glossaryNote')}
                className='h-8 text-[11px]'
                onChange={(event) =>
                  updateGlossary(index, { note: event.currentTarget.value || null })
                }
              />
              <Button
                type='button'
                variant='ghost'
                size='icon-sm'
                aria-label={t('settings.pipeline.translator.removeTerm')}
                title={t('settings.pipeline.translator.removeTerm')}
                className='shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive'
                onClick={() =>
                  onChange({
                    ...value,
                    glossary: glossary.filter((_, candidate) => candidate !== index),
                  })
                }
              >
                <Trash2 className='size-3.5' />
              </Button>
            </div>
          ))}
          <div>
            <Button
              type='button'
              variant='outline'
              size='xs'
              className='h-7 gap-1 px-2.5 text-[11px]'
              onClick={() =>
                onChange({
                  ...value,
                  glossary: [...glossary, { source: '', target: '', note: null }],
                })
              }
            >
              <Plus className='size-3' />
              <span>{t('settings.pipeline.translator.addTerm')}</span>
            </Button>
          </div>
        </div>
      </PreferenceRow>
    </>
  )
}
