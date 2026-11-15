'use client'

import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'

import { TitleBar } from '@/components/app/TitleBar'
import { Editor } from '@/components/editor/Editor'
import { SettingsPage } from '@/components/preferences/SettingsPage'
import { StartView } from '@/components/start/StartView'
import { usePages, useProject } from '@/lib/queries'
import { useKoharuStore } from '@/lib/store'
import { cn } from '@koharu/ui/lib/utils'

export function KoharuApp() {
  const { t } = useTranslation()
  const projectQuery = useProject()
  const project = projectQuery.data
  const hasProject = project !== undefined && project !== null
  const pagesQuery = usePages(hasProject)
  const pages = pagesQuery.data
  const projectFetching = projectQuery.isFetching
  const pagesFetching = pagesQuery.isFetching
  const selectedPages = useKoharuStore((state) => state.selectedPages)
  const selectPages = useKoharuStore((state) => state.selectPages)
  const selectLayers = useKoharuStore((state) => state.selectLayers)
  const settingsOpen = useKoharuStore((state) => state.settingsOpen)
  const activePage = project?.active_page
  const editorOpen = hasProject && !settingsOpen

  useEffect(() => {
    if (project === undefined) return
    if (project === null) {
      if (selectedPages.length > 0) {
        selectPages([])
        selectLayers([])
      }
      return
    }
    if (pages === undefined) {
      if (selectedPages.length === 0 && activePage && !projectFetching) {
        selectPages([activePage])
        selectLayers([])
      }
      return
    }
    if (projectFetching || pagesFetching) return

    const validIds = new Set(pages.map((page) => page.id))
    const validSelected = selectedPages.filter((id) => validIds.has(id))
    const next =
      validSelected.length > 0
        ? validSelected
        : activePage && validIds.has(activePage)
          ? [activePage]
          : pages[0]
            ? [pages[0].id]
            : []
    if (
      next.length === selectedPages.length &&
      next.every((id, index) => id === selectedPages[index])
    ) {
      return
    }
    selectPages(next)
    if (validSelected.length === 0) {
      selectLayers([])
    }
  }, [
    activePage,
    pages,
    pagesFetching,
    project,
    projectFetching,
    selectLayers,
    selectPages,
    selectedPages,
  ])

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
