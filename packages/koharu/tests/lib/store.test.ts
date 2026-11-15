import { describe, expect, it } from 'vitest'

import { receiveDownload, reconcileSelectedPages, useKoharuStore } from '@/lib/store'

describe('editor state', () => {
  it('retains failed downloads until they are dismissed', () => {
    useKoharuStore.setState({ downloads: {} })
    receiveDownload({
      state: { Failed: 'network unavailable' },
      id: 9,
      name: 'weights.bin',
      completed: 0,
      total: 0,
    })
    expect(useKoharuStore.getState().downloads[9]).toMatchObject({
      state: { Failed: 'network unavailable' },
      name: 'weights.bin',
    })

    useKoharuStore.getState().dismissDownload(9)
    expect(useKoharuStore.getState().downloads[9]).toBeUndefined()
  })

  it('defaults paint brush color to pure white (#FFFFFF)', () => {
    expect(useKoharuStore.getState().brush.color).toBe('#FFFFFF')
  })

  it('prunes deleted page IDs and falls back to the newly imported active page', () => {
    expect(reconcileSelectedPages(['deleted-page'], [], null)).toEqual([])
    expect(reconcileSelectedPages(['deleted-page'], [{ id: 'new-page' }], 'new-page')).toEqual([
      'new-page',
    ])
    expect(
      reconcileSelectedPages(
        ['deleted-page', 'kept-page'],
        [{ id: 'kept-page' }, { id: 'other-page' }],
        'other-page',
      ),
    ).toEqual(['kept-page'])
  })

  it('persists custom single-key shortcuts to localStorage', () => {
    useKoharuStore.getState().setShortcut('draw', 'P')
    expect(useKoharuStore.getState().shortcuts.draw).toBe('p')
    expect(JSON.parse(window.localStorage.getItem('koharu.shortcuts') ?? '{}')).toMatchObject({
      draw: 'p',
    })
    useKoharuStore.getState().setShortcut('draw', 'b')
  })
})
