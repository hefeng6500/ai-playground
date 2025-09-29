import React, { useState, useEffect } from 'react'
import { 
  Save, 
  RefreshCw, 
  TestTube, 
  AlertCircle, 
  CheckCircle, 
  Settings,
  Eye,
  EyeOff
} from 'lucide-react'
import { useConfigStore } from '../store'
import { configApi, SystemConfig } from '../services/api'
import { toast } from 'sonner'

interface ConfigFormData {
  openai_api_key: string
  openai_base_url: string
  openai_model: string
  max_tokens: number
  temperature: number
  app_name: string
  app_version: string
  debug_mode: boolean
  log_level: string
}

const ConfigPage: React.FC = () => {
  const {
    config,
    isLoading,
    error,
    setConfig,
    setLoading,
    setError
  } = useConfigStore()

  const [formData, setFormData] = useState<ConfigFormData>({
    openai_api_key: '',
    openai_base_url: '',
    openai_model: '',
    max_tokens: 1000,
    temperature: 0.7,
    app_name: '',
    app_version: '',
    debug_mode: false,
    log_level: 'INFO'
  })
  
  const [showApiKey, setShowApiKey] = useState(false)
  const [testResult, setTestResult] = useState<string>('')
  const [testLoading, setTestLoading] = useState(false)
  const [availableModels, setAvailableModels] = useState<string[]>([])
  const [hasChanges, setHasChanges] = useState(false)

  // 加载配置
  const loadConfig = async () => {
    setLoading(true)
    try {
      const configData = await configApi.getConfig()
      setConfig(configData)
      setFormData({
        openai_api_key: configData.openai_api_key || '',
        openai_base_url: configData.openai_base_url || '',
        openai_model: configData.openai_model || '',
        max_tokens: configData.max_tokens || 1000,
        temperature: configData.temperature || 0.7,
        app_name: configData.app_name || '',
        app_version: configData.app_version || '',
        debug_mode: configData.debug_mode || false,
        log_level: configData.log_level || 'INFO'
      })
      setHasChanges(false)
    } catch (error) {
      console.error('Failed to load config:', error)
      setError('加载配置失败')
      toast.error('加载配置失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载可用模型
  const loadAvailableModels = async () => {
    try {
      const models = await configApi.getAvailableModels()
      setAvailableModels(models)
    } catch (error) {
      console.error('Failed to load models:', error)
      toast.error('加载可用模型失败')
    }
  }

  // 保存配置
  const handleSave = async () => {
    setLoading(true)
    try {
      const updatedConfig = await configApi.updateConfig(formData)
      setConfig(updatedConfig)
      setHasChanges(false)
      toast.success('配置保存成功')
    } catch (error) {
      console.error('Failed to save config:', error)
      toast.error('保存配置失败')
    } finally {
      setLoading(false)
    }
  }

  // 重置配置
  const handleReset = async () => {
    if (!window.confirm('确定要重置所有配置为默认值吗？此操作不可撤销。')) {
      return
    }
    
    setLoading(true)
    try {
      const defaultConfig = await configApi.resetConfig()
      setConfig(defaultConfig)
      setFormData({
        openai_api_key: defaultConfig.openai_api_key || '',
        openai_base_url: defaultConfig.openai_base_url || '',
        openai_model: defaultConfig.openai_model || '',
        max_tokens: defaultConfig.max_tokens || 1000,
        temperature: defaultConfig.temperature || 0.7,
        app_name: defaultConfig.app_name || '',
        app_version: defaultConfig.app_version || '',
        debug_mode: defaultConfig.debug_mode || false,
        log_level: defaultConfig.log_level || 'INFO'
      })
      setHasChanges(false)
      toast.success('配置已重置为默认值')
    } catch (error) {
      console.error('Failed to reset config:', error)
      toast.error('重置配置失败')
    } finally {
      setLoading(false)
    }
  }

  // 测试配置
  const handleTest = async () => {
    setTestLoading(true)
    setTestResult('')
    try {
      const result = await configApi.testConfig()
      setTestResult(result.message || '配置测试成功')
      toast.success('配置测试完成')
    } catch (error) {
      console.error('Failed to test config:', error)
      setTestResult('配置测试失败')
      toast.error('配置测试失败')
    } finally {
      setTestLoading(false)
    }
  }

  // 处理表单变化
  const handleFormChange = (field: keyof ConfigFormData, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    setHasChanges(true)
  }

  // 组件挂载时加载数据
  useEffect(() => {
    loadConfig()
    loadAvailableModels()
  }, [])

  const logLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

  return (
    <div className="flex flex-col h-full bg-background">
      {/* 页面头部 */}
      <div className="flex-shrink-0 bg-card border-b border-border p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-foreground">系统配置</h2>
            <p className="text-muted-foreground mt-1">
              管理系统参数和OpenAI集成配置
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={handleTest}
              disabled={testLoading || isLoading}
              className="btn btn-ghost"
            >
              {testLoading ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <TestTube className="h-4 w-4 mr-2" />
              )}
              测试配置
            </button>
            <button
              onClick={handleReset}
              disabled={isLoading}
              className="btn btn-ghost text-destructive hover:text-destructive"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              重置
            </button>
            <button
              onClick={handleSave}
              disabled={!hasChanges || isLoading}
              className="btn btn-primary"
            >
              {isLoading ? (
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Save className="h-4 w-4 mr-2" />
              )}
              保存配置
            </button>
          </div>
        </div>
        
        {/* 测试结果 */}
        {testResult && (
          <div className={`
            mt-4 p-3 rounded-lg flex items-center space-x-2
            ${testResult.includes('成功') ? 'bg-green-50 text-green-700 border border-green-200' : 'bg-red-50 text-red-700 border border-red-200'}
          `}>
            {testResult.includes('成功') ? (
              <CheckCircle className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            <span className="text-sm">{testResult}</span>
          </div>
        )}
      </div>

      {/* 配置表单 */}
      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">加载中...</p>
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto space-y-8">
            {/* OpenAI 配置 */}
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Settings className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold text-foreground">OpenAI 配置</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-foreground mb-2">
                    API Key *
                  </label>
                  <div className="relative">
                    <input
                      type={showApiKey ? 'text' : 'password'}
                      value={formData.openai_api_key}
                      onChange={(e) => handleFormChange('openai_api_key', e.target.value)}
                      className="input w-full pr-10"
                      placeholder="输入OpenAI API Key"
                    />
                    <button
                      type="button"
                      onClick={() => setShowApiKey(!showApiKey)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Base URL
                  </label>
                  <input
                    type="url"
                    value={formData.openai_base_url}
                    onChange={(e) => handleFormChange('openai_base_url', e.target.value)}
                    className="input w-full"
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    模型
                  </label>
                  <select
                    value={formData.openai_model}
                    onChange={(e) => handleFormChange('openai_model', e.target.value)}
                    className="select w-full"
                  >
                    <option value="">选择模型</option>
                    {availableModels.map(model => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                  {availableModels.length === 0 && (
                    <p className="text-xs text-muted-foreground mt-1">
                      请先配置API Key以加载可用模型
                    </p>
                  )}
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    最大Token数
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="32000"
                    value={formData.max_tokens}
                    onChange={(e) => handleFormChange('max_tokens', parseInt(e.target.value))}
                    className="input w-full"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    Temperature (0-2)
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="2"
                    step="0.1"
                    value={formData.temperature}
                    onChange={(e) => handleFormChange('temperature', parseFloat(e.target.value))}
                    className="input w-full"
                  />
                  <p className="text-xs text-muted-foreground mt-1">
                    值越高输出越随机，值越低输出越确定
                  </p>
                </div>
              </div>
            </div>

            {/* 应用配置 */}
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-6">
                <Settings className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold text-foreground">应用配置</h3>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    应用名称
                  </label>
                  <input
                    type="text"
                    value={formData.app_name}
                    onChange={(e) => handleFormChange('app_name', e.target.value)}
                    className="input w-full"
                    placeholder="LangChain Knowledge Base"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    应用版本
                  </label>
                  <input
                    type="text"
                    value={formData.app_version}
                    onChange={(e) => handleFormChange('app_version', e.target.value)}
                    className="input w-full"
                    placeholder="1.0.0"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    日志级别
                  </label>
                  <select
                    value={formData.log_level}
                    onChange={(e) => handleFormChange('log_level', e.target.value)}
                    className="select w-full"
                  >
                    {logLevels.map(level => (
                      <option key={level} value={level}>
                        {level}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="debug_mode"
                    checked={formData.debug_mode}
                    onChange={(e) => handleFormChange('debug_mode', e.target.checked)}
                    className="rounded border-input"
                  />
                  <label htmlFor="debug_mode" className="text-sm text-foreground">
                    调试模式
                  </label>
                </div>
              </div>
            </div>

            {/* 当前配置信息 */}
            {config && (
              <div className="card p-6">
                <h3 className="text-lg font-semibold text-foreground mb-4">当前配置信息</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">最后更新:</span>
                    <span className="ml-2 text-foreground">
                      {config.updated_at ? new Date(config.updated_at).toLocaleString() : '未知'}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">配置状态:</span>
                    <span className={`ml-2 badge ${
                      config.openai_api_key ? 'badge-default' : 'badge-secondary'
                    }`}>
                      {config.openai_api_key ? '已配置' : '未配置'}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">当前模型:</span>
                    <span className="ml-2 text-foreground">
                      {config.openai_model || '未设置'}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">调试模式:</span>
                    <span className={`ml-2 badge ${
                      config.debug_mode ? 'badge-default' : 'badge-secondary'
                    }`}>
                      {config.debug_mode ? '开启' : '关闭'}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* 配置说明 */}
            <div className="card p-6 bg-muted">
              <h3 className="text-lg font-semibold text-foreground mb-4">配置说明</h3>
              <div className="space-y-3 text-sm text-muted-foreground">
                <div>
                  <strong className="text-foreground">API Key:</strong> 
                  OpenAI API密钥，用于访问OpenAI服务。请确保密钥有效且有足够的配额。
                </div>
                <div>
                  <strong className="text-foreground">Base URL:</strong> 
                  OpenAI API的基础URL，默认为官方地址。如使用代理服务，请修改此项。
                </div>
                <div>
                  <strong className="text-foreground">模型:</strong> 
                  选择要使用的OpenAI模型。不同模型有不同的能力和价格。
                </div>
                <div>
                  <strong className="text-foreground">最大Token数:</strong> 
                  单次请求的最大token数量，包括输入和输出。
                </div>
                <div>
                  <strong className="text-foreground">Temperature:</strong> 
                  控制输出的随机性。0表示确定性输出，2表示高度随机。
                </div>
                <div>
                  <strong className="text-foreground">调试模式:</strong> 
                  开启后会输出更详细的日志信息，便于问题排查。
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ConfigPage