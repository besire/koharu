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

  it('prunes deleted page IDs and falls back to the active page when importing fresh pages', () => {
    const pageB = {
      id: 'page-b',
      name: 'b.png',
      size: { width: 100, height: 100 },
      background: null,
    }
    const pageC = {
      id: 'page-c',
      name: 'c.png',
      size: { width: 100, height: 100 },
      background: null,
    }

    expect(reconcileSelectedPages(['deleted-page'], [], null)).toEqual([])
    expect(reconcileSelectedPages(['deleted-page'], [pageB], 'page-b')).toEqual(['page-b'])
    expect(reconcileSelectedPages(['deleted-page', 'page-c'], [pageB, pageC], 'page-b')).toEqual([
      'page-c',
    ])
  })
})
