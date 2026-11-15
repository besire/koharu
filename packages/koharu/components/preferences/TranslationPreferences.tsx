'use client'

import { ChevronDown, Plus, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ModelPicker } from '@/components/controls/ModelPicker'
import { vendorName } from '@/components/preferences/models'
import {
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { modelName, orderedLanguageChoices, REASONING, vendorOf } from '@/lib/translation'
import type { GlossaryEntry, Model, TranslatorConfig } from '@koharu/ipc/protocol'
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

/** Whether the prompt has `{target}`, which the translator refuses one without. */
export function promptNamesTarget(value: TranslatorConfig): boolean {
  return (value.prompt ?? '').includes('{target}')
}

/** The `[pipeline.translator]` section: the model, the target language, the system prompt with
 * `{target}` where the target's name goes, whether the model sees each page, and how much it
 * reasons. */
export function TranslationPreferences({
  value,
  models,
  onChange,
}: {
  value: TranslatorConfig
  models: Model[]
  onChange: (value: TranslatorConfig) => void
}) {
  const { t, i18n } = useTranslation()
  const [modelOpen, setModelOpen] = useState(false)
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
  const model = value.model ?? ''
  const reasoning = value.reasoning ?? 'default'
  const presets = value.presets ?? []
  const glossary = value.glossary ?? []
  const activePreset = presets.find((preset) => preset.prompt === (value.prompt ?? ''))?.name ?? ''

  const updateGlossary = (index: number, patch: Partial<GlossaryEntry>) => {
    const next = glossary.map((entry, candidate) =>
      candidate === index ? { ...entry, ...patch } : entry,
    )
    onChange({ ...value, glossary: next })
  }

  return (
    <PreferencePage
      title={t('settings.translation.title')}
      description={t('settings.translation.description')}
    >
      <PreferenceSection
        title={t('settings.translation.model')}
        description={t('settings.translation.modelDescription')}
      >
        <PreferenceRow
          title={t('settings.translation.translationModel')}
          description={t('settings.translation.translationModelDescription')}
        >
          <Popover open={modelOpen} onOpenChange={setModelOpen}>
            <PopoverTrigger
              type='button'
              aria-label={t('settings.translation.translationModel')}
              className='flex h-9 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-transparent px-2.5 text-[11px] transition-colors outline-none hover:bg-foreground/[0.03] focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50'
            >
              <span className='flex min-w-0 flex-1 items-center gap-2 text-left'>
                {model && (
                  <Badge variant='outline' className='shrink-0 px-1.5 py-0 text-[9px] font-medium'>
                    {vendorName(vendorOf(model), t)}
                  </Badge>
                )}
                <span className='truncate'>
                  {modelName(models, model) || t('inference.noModel')}
                </span>
              </span>
              <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
            </PopoverTrigger>
            <PopoverContent
              align='start'
              sideOffset={4}
              className='w-(--anchor-width) min-w-64 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
            >
              <ModelPicker
                value={model}
                models={models}
                onBack={() => setModelOpen(false)}
                onSelect={(reference) => {
                  onChange({ ...value, model: reference })
                  setModelOpen(false)
                }}
              />
            </PopoverContent>
          </Popover>
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.translation.vision')}
          description={t('settings.translation.visionDescription')}
        >
          <Switch
            aria-label={t('settings.translation.vision')}
            checked={value.vision ?? false}
            onCheckedChange={(vision) => onChange({ ...value, vision })}
          />
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.translation.reasoning')}
          description={t('settings.translation.reasoningDescription')}
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
              aria-label={t('settings.translation.reasoning')}
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
      </PreferenceSection>

      <PreferenceSection title={t('settings.translation.output')}>
        <PreferenceRow
          title={t('model.sourceLanguage')}
          description={t('settings.translation.sourceLanguageDescription')}
        >
          <Select
            value={value.source ?? 'auto'}
            items={sourceItems}
            onValueChange={(source) =>
              source && onChange({ ...value, source: source === 'auto' ? null : source })
            }
          >
            <SelectTrigger
              aria-label={t('model.sourceLanguage')}
              className='h-8 w-full text-[11px]'
            >
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
          description={t('settings.translation.targetLanguageDescription')}
        >
          <Select
            value={value.target ?? 'en'}
            items={Object.fromEntries(
              languageChoices.map((language) => [language.tag, language.name]),
            )}
            onValueChange={(target) => target && onChange({ ...value, target })}
          >
            <SelectTrigger
              aria-label={t('model.targetLanguage')}
              className='h-8 w-full text-[11px]'
            >
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
          title={t('settings.translation.prompt')}
          description={t('settings.translation.promptDescription')}
          align='start'
        >
          <div className='grid gap-1.5'>
            {presets.length > 0 && (
              <div className='flex items-center gap-1.5'>
                <Select
                  value={activePreset || '__custom__'}
                  items={Object.fromEntries([
                    ['__custom__', t('settings.translation.presets')],
                    ...presets.map((preset) => [preset.name, preset.name]),
                  ])}
                  onValueChange={(selected) => {
                    const match = presets.find((preset) => preset.name === selected)
                    if (match) onChange({ ...value, prompt: match.prompt })
                  }}
                >
                  <SelectTrigger
                    aria-label={t('settings.translation.presets')}
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
                  aria-label={t('settings.translation.deletePreset')}
                  title={t('settings.translation.deletePreset')}
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
              aria-label={t('settings.translation.prompt')}
              aria-invalid={!promptNamesTarget(value)}
              value={value.prompt ?? ''}
              className='field-sizing-fixed min-h-32 resize-y overflow-y-auto text-[12px] leading-5'
              onChange={(event) => onChange({ ...value, prompt: event.currentTarget.value })}
            />
            {!promptNamesTarget(value) && (
              <p className='text-[10px] text-destructive'>
                {t('settings.translation.promptNeedsTarget')}
              </p>
            )}
            <div className='flex items-center gap-1.5'>
              <Input
                value={presetName}
                aria-label={t('settings.translation.presetName')}
                placeholder={t('settings.translation.presetName')}
                className='h-7 flex-1 text-[11px]'
                onChange={(event) => setPresetName(event.currentTarget.value)}
              />
              <Button
                type='button'
                variant='outline'
                size='xs'
                disabled={!presetName.trim() || !promptNamesTarget(value)}
                className='h-7 shrink-0 px-2.5 text-[11px]'
                onClick={() => {
                  const name = presetName.trim()
                  if (!name) return
                  const prompt = value.prompt ?? ''
                  const nextPresets = presets.some((preset) => preset.name === name)
                    ? presets.map((preset) => (preset.name === name ? { name, prompt } : preset))
                    : [...presets, { name, prompt }]
                  onChange({ ...value, presets: nextPresets })
                  setPresetName('')
                }}
              >
                {t('settings.translation.savePreset')}
              </Button>
            </div>
          </div>
        </PreferenceRow>
      </PreferenceSection>

      <PreferenceSection
        title={t('settings.translation.glossary')}
        description={t('settings.translation.glossaryDescription')}
      >
        <div className='grid gap-2 py-3.5'>
          {glossary.map((entry, index) => (
            <div
              key={index}
              className='grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)_auto] items-center gap-2'
            >
              <Input
                value={entry.source}
                aria-label={t('settings.translation.glossarySource')}
                placeholder={t('settings.translation.glossarySource')}
                className='h-8 text-[11px]'
                onChange={(event) => updateGlossary(index, { source: event.currentTarget.value })}
              />
              <Input
                value={entry.target}
                aria-label={t('settings.translation.glossaryTarget')}
                placeholder={t('settings.translation.glossaryTarget')}
                className='h-8 text-[11px]'
                onChange={(event) => updateGlossary(index, { target: event.currentTarget.value })}
              />
              <Input
                value={entry.note ?? ''}
                aria-label={t('settings.translation.glossaryNote')}
                placeholder={t('settings.translation.glossaryNote')}
                className='h-8 text-[11px]'
                onChange={(event) =>
                  updateGlossary(index, { note: event.currentTarget.value || null })
                }
              />
              <Button
                type='button'
                variant='ghost'
                size='icon-sm'
                aria-label={t('settings.translation.removeTerm')}
                title={t('settings.translation.removeTerm')}
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
              <span>{t('settings.translation.addTerm')}</span>
            </Button>
          </div>
        </div>
      </PreferenceSection>
    </PreferencePage>
  )
}
