import React, { useState, useEffect } from 'react'
import { 
  Search, 
  Filter, 
  Download, 
  Trash2, 
  RefreshCw,
  Calendar,
  AlertCircle,
  Info,
  AlertTriangle,
  XCircle,
  Bug,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { logApi, LogEntry } from '../services/api'
import { toast } from 'sonner'

interface LogFilters {
  level: string
  startDate: string
  endDate: string
  keyword: string
}

interface LogStats {
  total: number
  by_level: Record<string, number>
  by_date: Record<string, number>
}

const LogsPage: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [stats, setStats] = useState<LogStats | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [filters, setFilters] = useState<LogFilters>({
    level: '',
    startDate: '',
    endDate: '',
    keyword: ''
  })
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize] = useState(50)
  const [totalLogs, setTotalLogs] = useState(0)
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set())
  const [showFilters, setShowFilters] = useState(false)

  // 日志级别配置
  const logLevels = [
    { value: '', label: '全部级别' },
    { value: 'DEBUG', label: 'DEBUG', color: 'text-gray-500', bg: 'bg-gray-100' },
    { value: 'INFO', label: 'INFO', color: 'text-blue-600', bg: 'bg-blue-100' },
    { value: 'WARNING', label: 'WARNING', color: 'text-yellow-600', bg: 'bg-yellow-100' },
    { value: 'ERROR', label: 'ERROR', color: 'text-red-600', bg: 'bg-red-100' },
    { value: 'CRITICAL', label: 'CRITICAL', color: 'text-red-800', bg: 'bg-red-200' }
  ]

  // 获取日志级别样式
  const getLevelStyle = (level: string) => {
    const config = logLevels.find(l => l.value === level)
    return config ? { color: config.color, bg: config.bg } : { color: 'text-gray-500', bg: 'bg-gray-100' }
  }

  // 获取日志级别图标
  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'DEBUG':
        return <Bug className="h-4 w-4" />
      case 'INFO':
        return <Info className="h-4 w-4" />
      case 'WARNING':
        return <AlertTriangle className="h-4 w-4" />
      case 'ERROR':
      case 'CRITICAL':
        return <XCircle className="h-4 w-4" />
      default:
        return <AlertCircle className="h-4 w-4" />
    }
  }

  // 加载日志数据
  const loadLogs = async (page = 1) => {
    setIsLoading(true)
    try {
      const response = await logApi.getLogs({
        level: filters.level || undefined,
        start_date: filters.startDate || undefined,
        end_date: filters.endDate || undefined,
        keyword: filters.keyword || undefined,
        page,
        page_size: pageSize
      })
      
      setLogs(response.logs)
      setTotalLogs(response.total)
      setCurrentPage(page)
    } catch (error) {
      console.error('Failed to load logs:', error)
      toast.error('加载日志失败')
    } finally {
      setIsLoading(false)
    }
  }

  // 加载日志统计
  const loadStats = async () => {
    try {
      const statsData = await logApi.getLogStats({
        start_date: filters.startDate || undefined,
        end_date: filters.endDate || undefined
      })
      setStats(statsData)
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  // 清理日志
  const handleClearLogs = async () => {
    const confirmMessage = filters.level || filters.startDate || filters.endDate
      ? '确定要清理符合当前筛选条件的日志吗？'
      : '确定要清理所有日志吗？此操作不可撤销。'
    
    if (!window.confirm(confirmMessage)) {
      return
    }
    
    try {
      await logApi.clearLogs({
        level: filters.level || undefined,
        before_date: filters.endDate || undefined
      })
      
      toast.success('日志清理成功')
      loadLogs(1)
      loadStats()
    } catch (error) {
      console.error('Failed to clear logs:', error)
      toast.error('清理日志失败')
    }
  }

  // 导出日志
  const handleExportLogs = async (format: 'json' | 'csv') => {
    try {
      const blob = await logApi.exportLogs({
        level: filters.level || undefined,
        start_date: filters.startDate || undefined,
        end_date: filters.endDate || undefined,
        keyword: filters.keyword || undefined,
        format
      })
      
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `logs_${new Date().toISOString().split('T')[0]}.${format}`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      toast.success(`日志已导出为 ${format.toUpperCase()} 格式`)
    } catch (error) {
      console.error('Failed to export logs:', error)
      toast.error('导出日志失败')
    }
  }

  // 切换日志展开状态
  const toggleLogExpanded = (logId: string) => {
    const newExpanded = new Set(expandedLogs)
    if (newExpanded.has(logId)) {
      newExpanded.delete(logId)
    } else {
      newExpanded.add(logId)
    }
    setExpandedLogs(newExpanded)
  }

  // 应用筛选
  const applyFilters = () => {
    setCurrentPage(1)
    loadLogs(1)
    loadStats()
  }

  // 重置筛选
  const resetFilters = () => {
    setFilters({
      level: '',
      startDate: '',
      endDate: '',
      keyword: ''
    })
    setCurrentPage(1)
  }

  // 分页计算
  const totalPages = Math.ceil(totalLogs / pageSize)
  const startIndex = (currentPage - 1) * pageSize + 1
  const endIndex = Math.min(currentPage * pageSize, totalLogs)

  // 组件挂载时加载数据
  useEffect(() => {
    loadLogs()
    loadStats()
  }, [])

  // 筛选条件变化时重新加载
  useEffect(() => {
    if (filters.level || filters.startDate || filters.endDate || filters.keyword) {
      const timer = setTimeout(() => {
        applyFilters()
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [filters])

  return (
    <div className="flex flex-col h-full bg-background">
      {/* 页面头部 */}
      <div className="flex-shrink-0 bg-card border-b border-border p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-foreground">系统日志</h2>
            <p className="text-muted-foreground mt-1">
              查看和管理系统运行日志
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="btn btn-ghost"
            >
              <Filter className="h-4 w-4 mr-2" />
              筛选
              {showFilters ? <ChevronUp className="h-4 w-4 ml-1" /> : <ChevronDown className="h-4 w-4 ml-1" />}
            </button>
            <div className="relative">
              <button className="btn btn-ghost peer">
                <Download className="h-4 w-4 mr-2" />
                导出
              </button>
              <div className="absolute right-0 top-full mt-1 bg-card border border-border rounded-lg shadow-lg opacity-0 invisible peer-hover:opacity-100 peer-hover:visible hover:opacity-100 hover:visible transition-all z-10">
                <div className="p-2 space-y-1">
                  <button
                    onClick={() => handleExportLogs('json')}
                    className="block w-full text-left px-3 py-2 text-sm hover:bg-muted rounded"
                  >
                    导出为 JSON
                  </button>
                  <button
                    onClick={() => handleExportLogs('csv')}
                    className="block w-full text-left px-3 py-2 text-sm hover:bg-muted rounded"
                  >
                    导出为 CSV
                  </button>
                </div>
              </div>
            </div>
            <button
              onClick={handleClearLogs}
              className="btn btn-ghost text-destructive hover:text-destructive"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              清理
            </button>
            <button
              onClick={() => {
                loadLogs(currentPage)
                loadStats()
              }}
              disabled={isLoading}
              className="btn btn-ghost"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              刷新
            </button>
          </div>
        </div>

        {/* 统计信息 */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
            <div className="bg-muted p-3 rounded-lg">
              <div className="text-sm text-muted-foreground">总计</div>
              <div className="text-lg font-semibold text-foreground">{stats.total}</div>
            </div>
            {logLevels.slice(1).map(level => {
              const count = stats.by_level[level.value] || 0
              const style = getLevelStyle(level.value)
              return (
                <div key={level.value} className="bg-muted p-3 rounded-lg">
                  <div className="text-sm text-muted-foreground">{level.label}</div>
                  <div className={`text-lg font-semibold ${style.color}`}>{count}</div>
                </div>
              )
            })}
          </div>
        )}

        {/* 筛选器 */}
        {showFilters && (
          <div className="bg-muted p-4 rounded-lg space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  日志级别
                </label>
                <select
                  value={filters.level}
                  onChange={(e) => setFilters(prev => ({ ...prev, level: e.target.value }))}
                  className="select w-full"
                >
                  {logLevels.map(level => (
                    <option key={level.value} value={level.value}>
                      {level.label}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  开始日期
                </label>
                <input
                  type="date"
                  value={filters.startDate}
                  onChange={(e) => setFilters(prev => ({ ...prev, startDate: e.target.value }))}
                  className="input w-full"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  结束日期
                </label>
                <input
                  type="date"
                  value={filters.endDate}
                  onChange={(e) => setFilters(prev => ({ ...prev, endDate: e.target.value }))}
                  className="input w-full"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-foreground mb-2">
                  关键词
                </label>
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <input
                    type="text"
                    value={filters.keyword}
                    onChange={(e) => setFilters(prev => ({ ...prev, keyword: e.target.value }))}
                    className="input pl-10 w-full"
                    placeholder="搜索日志内容"
                  />
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-end space-x-3">
              <button
                onClick={resetFilters}
                className="btn btn-ghost"
              >
                重置
              </button>
              <button
                onClick={applyFilters}
                className="btn btn-primary"
              >
                应用筛选
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 日志列表 */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">加载中...</p>
            </div>
          </div>
        ) : logs.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <p className="text-lg text-muted-foreground mb-2">暂无日志数据</p>
              <p className="text-sm text-muted-foreground">
                {Object.values(filters).some(v => v) ? '没有找到匹配的日志' : '系统还没有生成日志'}
              </p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {logs.map((log) => {
              const style = getLevelStyle(log.level)
              const isExpanded = expandedLogs.has(log.id)
              
              return (
                <div key={log.id} className="p-4 hover:bg-muted/50 transition-colors">
                  <div className="flex items-start space-x-3">
                    <div className={`flex items-center space-x-2 ${style.color}`}>
                      {getLevelIcon(log.level)}
                      <span className={`badge text-xs ${style.bg} ${style.color}`}>
                        {log.level}
                      </span>
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-sm text-muted-foreground">
                          {new Date(log.timestamp).toLocaleString()}
                        </div>
                        {log.message.length > 100 && (
                          <button
                            onClick={() => toggleLogExpanded(log.id)}
                            className="text-xs text-primary hover:underline"
                          >
                            {isExpanded ? '收起' : '展开'}
                          </button>
                        )}
                      </div>
                      
                      <div className="text-foreground">
                        {isExpanded || log.message.length <= 100 ? (
                          <pre className="whitespace-pre-wrap text-sm font-mono">
                            {log.message}
                          </pre>
                        ) : (
                          <p className="text-sm">
                            {log.message.substring(0, 100)}...
                          </p>
                        )}
                      </div>
                      
                      {log.module && (
                        <div className="mt-2">
                          <span className="badge badge-outline text-xs">
                            {log.module}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* 分页 */}
      {totalLogs > 0 && (
        <div className="flex-shrink-0 bg-card border-t border-border p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm text-muted-foreground">
              显示 {startIndex}-{endIndex} 条，共 {totalLogs} 条日志
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={() => loadLogs(currentPage - 1)}
                disabled={currentPage <= 1 || isLoading}
                className="btn btn-ghost btn-sm"
              >
                上一页
              </button>
              
              <div className="flex items-center space-x-1">
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  let pageNum
                  if (totalPages <= 5) {
                    pageNum = i + 1
                  } else if (currentPage <= 3) {
                    pageNum = i + 1
                  } else if (currentPage >= totalPages - 2) {
                    pageNum = totalPages - 4 + i
                  } else {
                    pageNum = currentPage - 2 + i
                  }
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => loadLogs(pageNum)}
                      disabled={isLoading}
                      className={`btn btn-sm ${
                        pageNum === currentPage ? 'btn-primary' : 'btn-ghost'
                      }`}
                    >
                      {pageNum}
                    </button>
                  )
                })}
              </div>
              
              <button
                onClick={() => loadLogs(currentPage + 1)}
                disabled={currentPage >= totalPages || isLoading}
                className="btn btn-ghost btn-sm"
              >
                下一页
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default LogsPage