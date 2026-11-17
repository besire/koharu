'use client'

import { ChevronLeft } from 'lucide-react'
import { useMemo } from 'react'
import { useTranslation } from 'react-i18next'

import { orderedLanguageChoices } from '@/lib/translation'
import { Button } from '@koharu/ui/components/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@koharu/ui/components/select'

/** The translator's source and target languages, saved as soon as they're picked. */
export function OutputPicker({
  sourceLanguage,
  targetLanguage,
  disabled = false,
  saving = false,
  onBack,
  onSourceChange,
  onChange,
}: {
  sourceLanguage: string | null
  targetLanguage: string
  disabled?: boolean
  saving?: boolean
  onBack: () => void
  onSourceChange: (sourceLanguage: string | null) => void
  onChange: (targetLanguage: string) => void
}) {
  const { t, i18n } = useTranslation()
  const languageChoices = useMemo(() => orderedLanguageChoices(i18n.language), [i18n.language])
  const sourceItems = useMemo(
    () =>
      Object.fromEntries([
        ['auto', t('model.autoDetect')],
        ...languageChoices.map((language) => [language.tag, language.name]),
      ]),
    [languageChoices, t],
  )

  return (
    <div className='min-w-0 overflow-hidden'>
      <div className='mb-1 flex h-7 items-center border-b border-border/60 px-0.5 pb-1'>
        <Button
          type='button'
          variant='ghost'
          size='icon-xs'
          aria-label={t('common.back')}
          className='rounded-md text-muted-foreground hover:bg-primary/10 hover:text-foreground'
          disabled={saving}
          onClick={onBack}
        >
          <ChevronLeft className='size-3.5' />
        </Button>
        <span className='ml-1 text-[11px] font-medium'>{t('outputPicker.title')}</span>
      </div>

      <div className='grid gap-2 p-1'>
        <label className='grid gap-1 text-[9px] text-muted-foreground'>
          {t('model.sourceLanguage')}
          <Select
            value={sourceLanguage ?? 'auto'}
            items={sourceItems}
            disabled={disabled || saving}
            onValueChange={(next) => {
              if (!next) return
              const resolved = next === 'auto' ? null : next
              if (resolved !== sourceLanguage) onSourceChange(resolved)
            }}
          >
            <SelectTrigger
              aria-label={t('outputPicker.sourceLanguage')}
              className='h-7 w-full text-[11px]'
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
        </label>
        <label className='grid gap-1 text-[9px] text-muted-foreground'>
          {t('model.targetLanguage')}
          <Select
            value={targetLanguage}
            items={Object.fromEntries(
              languageChoices.map((language) => [language.tag, language.name]),
            )}
            disabled={disabled || saving}
            onValueChange={(next) => next && next !== targetLanguage && onChange(next)}
          >
            <SelectTrigger
              aria-label={t('outputPicker.targetLanguage')}
              className='h-7 w-full text-[11px]'
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
        </label>
      </div>
    </div>
  )
}
