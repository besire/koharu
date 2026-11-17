'use client'

import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'

import { TitleBar } from '@/components/app/TitleBar'
import { Editor } from '@/components/editor/Editor'
import { SettingsPage } from '@/components/preferences/SettingsPage'
import { StartView } from '@/components/start/StartView'
import { useProject } from '@/lib/queries'
import { reconcileSelectedPages, useKoharuStore } from '@/lib/store'
import { cn } from '@koharu/ui/lib/utils'

export function KoharuApp() {
  const { t } = useTranslation()
  const project = useProject().data
  const selectedPages = useKoharuStore((state) => state.selectedPages)
  const selectPages = useKoharuStore((state) => state.selectPages)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const projectLoaded = project !== undefined
  const settingsOpen = useKoharuStore((state) => state.settingsOpen)
  const activePage = project?.page
  const pages = project?.pages
  const editorOpen = project !== undefined && project !== null && !settingsOpen

  useEffect(() => {
    if (!projectLoaded) return
    const next = reconcileSelectedPages(selectedPages, pages ?? [], activePage)
    const changed =
      next.length !== selectedPages.length || next.some((id, index) => id !== selectedPages[index])
    if (changed) {
      selectPages(next)
      if (selectedPages.length === 0 || next.length === 0) selectLayers([])
    }
  }, [activePage, pages, projectLoaded, selectLayers, selectPages, selectedPages])

  return (
    <div
      className={cn(
        'relative flex h-screen w-screen flex-col overflow-hidden text-foreground',
        editorOpen ? 'bg-transparent' : 'bg-[var(--surface-titlebar)]',
      )}
    >
      <TitleBar />
      {settingsOpen ? (
        <SettingsPage />
      ) : project === undefined ? (
        <main className='grid min-h-0 flex-1 place-items-center bg-[var(--surface-canvas)]'>
          <div className='flex items-center gap-3 text-[12px] text-muted-foreground'>
            <span className='size-2 rounded-full bg-primary' />
            {t('startup.opening')}
          </div>
        </main>
      ) : project === null ? (
        <StartView />
      ) : (
        <Editor />
      )}
    </div>
  )
}
