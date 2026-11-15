'use client'

import { ChevronDown, Plus, Trash2 } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'

import { ModelPicker } from '@/components/controls/ModelPicker'
import { GenerationPreferences } from '@/components/preferences/GenerationPreferences'
import {
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { modelKey, orderedLanguageChoices, providerName } from '@/lib/translation'
import type {
  LanguageChoice,
  Model,
  ProviderPreference,
  TranslationConfig as TranslationSettings,
} from '@koharu/bridge/protocol'
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
import { Textarea } from '@koharu/ui/components/textarea'

export function TranslationPreferences({
  value,
  modelChoices,
  providers,
  languages,
  onChange,
}: {
  value: TranslationSettings
  modelChoices: Model[]
  providers: ProviderPreference[]
  languages: LanguageChoice[]
  onChange: (value: TranslationSettings) => void
}) {
  const { t } = useTranslation()
  const [modelOpen, setModelOpen] = useState(false)
  const selected =
    modelChoices.find((candidate) => modelKey(candidate) === modelKey(value.model)) ?? null
  const current: Model = selected ?? {
    ...value.model,
    model: value.model.model ?? null,
    name: value.model.model ?? providerName(providers, value.model.provider),
    quantizations: [],
    vision: value.model.vision ?? false,
    reasoning: value.model.reasoning ?? false,
  }
  const choices = selected ? modelChoices : [current, ...modelChoices]
  const quantization =
    current.quantizations.find((quantization) => quantization.id === value.model.quantization) ??
    current.quantizations[0]
  const languageChoices = useMemo(() => orderedLanguageChoices(languages), [languages])
  const [presetName, setPresetName] = useState('')
  const [selectedPreset, setSelectedPreset] = useState('')
  const [glossarySource, setGlossarySource] = useState('')
  const [glossaryTarget, setGlossaryTarget] = useState('')
  const presets = value.instruction_presets ?? []
  const glossary = value.glossary ?? []
  const selectedExists = presets.some((preset) => preset.name === selectedPreset)

  const addGlossaryEntry = () => {
    const source = glossarySource.trim()
    const target = glossaryTarget.trim()
    if (!source || !target) return
    const next = [...glossary]
    const index = next.findIndex((entry) => entry.source === source)
    if (index >= 0) next[index] = { source, target }
    else next.push({ source, target })
    setGlossarySource('')
    setGlossaryTarget('')
    onChange({ ...value, glossary: next })
  }

  const removeGlossaryEntry = (source: string) => {
    onChange({
      ...value,
      glossary: glossary.filter((entry) => entry.source !== source),
    })
  }

  const savePreset = () => {
    const name = presetName.trim()
    if (!name) return
    const next = { name, instructions: value.instructions ?? '' }
    const instruction_presets = [...presets]
    const index = instruction_presets.findIndex((preset) => preset.name === name)
    if (index >= 0) instruction_presets[index] = next
    else instruction_presets.push(next)
    setSelectedPreset(name)
    setPresetName(name)
    onChange({ ...value, instruction_presets })
  }

  const loadPreset = (name: string) => {
    const preset = presets.find((candidate) => candidate.name === name)
    if (!preset) return
    setSelectedPreset(name)
    setPresetName(name)
    onChange({ ...value, instructions: preset.instructions || null })
  }

  const deletePreset = () => {
    if (!selectedExists) return
    onChange({
      ...value,
      instruction_presets: presets.filter((preset) => preset.name !== selectedPreset),
    })
    setSelectedPreset('')
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
              <span className='min-w-0 flex-1 text-left'>
                <ModelLabel
                  model={current}
                  providers={providers}
                  quantization={quantization?.name ?? value.model.quantization}
                />
              </span>
              <ChevronDown className='size-3.5 shrink-0 text-muted-foreground' />
            </PopoverTrigger>
            <PopoverContent
              align='start'
              sideOffset={4}
              className='w-(--anchor-width) min-w-64 gap-0 overflow-hidden rounded-xl border border-border/50 p-1 shadow-sm ring-0'
            >
              <ModelPicker
                value={value.model}
                models={choices}
                providers={providers}
                onBack={() => setModelOpen(false)}
                onSelect={(model) => {
                  onChange({
                    ...value,
                    model,
                  })
                  setModelOpen(false)
                }}
              />
            </PopoverContent>
          </Popover>
        </PreferenceRow>
      </PreferenceSection>

      <GenerationPreferences
        value={value.generation}
        onChange={(generation) => onChange({ ...value, generation })}
      />

      <PreferenceSection title={t('settings.translation.output')}>
        <PreferenceRow
          title={t('model.targetLanguage')}
          description={t('settings.translation.targetLanguageDescription')}
        >
          <Select
            value={value.target_language}
            items={Object.fromEntries(
              languageChoices.map((language) => [language.tag, language.name]),
            )}
            onValueChange={(target_language) =>
              target_language && onChange({ ...value, target_language })
            }
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
          title={t('model.instructions')}
          description={t('settings.translation.instructionsDescription')}
          align='start'
        >
          <Textarea
            aria-label={t('settings.translation.instructionsLabel')}
            value={value.instructions ?? ''}
            className='field-sizing-fixed min-h-24 resize-y overflow-y-auto text-[12px] leading-5'
            placeholder={t('settings.translation.instructionsPlaceholder')}
            onChange={(event) =>
              onChange({ ...value, instructions: event.currentTarget.value || null })
            }
          />
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.translation.savedPresets')}
          description={t('settings.translation.presetsDescription')}
          align='start'
        >
          <div className='grid gap-2'>
            <div className='flex gap-2'>
              <Input
                value={presetName}
                aria-label={t('settings.translation.presetName')}
                placeholder={t('settings.translation.presetNamePlaceholder')}
                className='h-8 min-w-0 flex-1 text-[12px]'
                onChange={(event) => setPresetName(event.currentTarget.value)}
              />
              <Button
                type='button'
                variant='outline'
                size='sm'
                className='h-8'
                disabled={!presetName.trim()}
                onClick={savePreset}
              >
                {t('settings.translation.savePreset')}
              </Button>
            </div>
            <div className='flex gap-2'>
              <Select
                value={selectedExists ? selectedPreset : ''}
                items={Object.fromEntries(presets.map((preset) => [preset.name, preset.name]))}
                onValueChange={(name) => name && loadPreset(name)}
              >
                <SelectTrigger
                  aria-label={t('settings.translation.savedPresets')}
                  className='h-8 min-w-0 flex-1 text-[11px]'
                  disabled={presets.length === 0}
                >
                  <SelectValue
                    placeholder={
                      presets.length === 0
                        ? t('settings.translation.noPresets')
                        : t('settings.translation.savedPresets')
                    }
                  />
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
                aria-label={t('settings.translation.deletePreset')}
                title={t('settings.translation.deletePreset')}
                className='shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive'
                disabled={!selectedExists}
                onClick={deletePreset}
              >
                <Trash2 className='size-3.5' />
              </Button>
            </div>
          </div>
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.translation.glossary')}
          description={t('settings.translation.glossaryDescription')}
          align='start'
        >
          <div className='grid gap-2'>
            <div className='flex gap-2'>
              <Input
                value={glossarySource}
                aria-label={t('settings.translation.glossarySource')}
                placeholder={t('settings.translation.glossarySourcePlaceholder')}
                className='h-8 min-w-0 flex-1 text-[12px]'
                onChange={(event) => setGlossarySource(event.currentTarget.value)}
              />
              <Input
                value={glossaryTarget}
                aria-label={t('settings.translation.glossaryTarget')}
                placeholder={t('settings.translation.glossaryTargetPlaceholder')}
                className='h-8 min-w-0 flex-1 text-[12px]'
                onChange={(event) => setGlossaryTarget(event.currentTarget.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    addGlossaryEntry()
                  }
                }}
              />
              <Button
                type='button'
                variant='outline'
                size='icon-sm'
                aria-label={t('settings.translation.addGlossaryEntry')}
                title={t('settings.translation.addGlossaryEntry')}
                disabled={!glossarySource.trim() || !glossaryTarget.trim()}
                onClick={addGlossaryEntry}
              >
                <Plus className='size-3.5' />
              </Button>
            </div>
            {glossary.length > 0 && (
              <ul className='divide-y divide-border/60 rounded-lg border border-border/60'>
                {glossary.map((entry) => (
                  <li
                    key={entry.source}
                    className='flex min-w-0 items-center gap-2 px-2.5 py-1.5 text-[11px]'
                  >
                    <span className='min-w-0 flex-1 truncate font-medium'>{entry.source}</span>
                    <span className='shrink-0 text-muted-foreground'>→</span>
                    <span className='min-w-0 flex-1 truncate'>{entry.target}</span>
                    <Button
                      type='button'
                      variant='ghost'
                      size='icon-xs'
                      aria-label={t('settings.translation.removeGlossaryEntry', {
                        source: entry.source,
                      })}
                      className='shrink-0 text-muted-foreground hover:bg-destructive/10 hover:text-destructive'
                      onClick={() => removeGlossaryEntry(entry.source)}
                    >
                      <Trash2 className='size-3' />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </PreferenceRow>
      </PreferenceSection>
    </PreferencePage>
  )
}

function ModelLabel({
  model,
  providers,
  quantization,
}: {
  model: Model
  providers: ProviderPreference[]
  quantization: string | null | undefined
}) {
  return (
    <span className='flex min-w-0 items-center gap-2'>
      <Badge variant='outline' className='shrink-0 px-1.5 py-0 text-[9px] font-medium'>
        {providerName(providers, model.provider)}
      </Badge>
      <span className='truncate'>
        {model.name}
        {quantization && <span className='text-muted-foreground'> · {quantization}</span>}
      </span>
    </span>
  )
}
