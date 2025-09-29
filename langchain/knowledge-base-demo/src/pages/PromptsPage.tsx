import React, { useState, useEffect } from 'react'
import { 
  Plus, 
  Edit, 
  Trash2, 
  Search, 
  Filter, 
  Eye, 
  Copy, 
  Check,
  X,
  Save,
  TestTube
} from 'lucide-react'
import { usePromptStore, generateId } from '../store'
import { promptApi, PromptTemplate, PromptTemplateCreate, PromptTemplateUpdate } from '../services/api'
import { toast } from 'sonner'

interface PromptFormData {
  name: string
  description: string
  template: string
  category: string
  is_active: boolean
  variables: string[]
}

const PromptsPage: React.FC = () => {
  const {
    prompts,
    selectedPrompt,
    isLoading,
    error,
    setPrompts,
    setSelectedPrompt,
    addPrompt,
    updatePrompt,
    removePrompt,
    setLoading,
    setError
  } = usePromptStore()

  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState('')
  const [showActiveOnly, setShowActiveOnly] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showEditModal, setShowEditModal] = useState(false)
  const [showViewModal, setShowViewModal] = useState(false)
  const [categories, setCategories] = useState<string[]>([])
  const [formData, setFormData] = useState<PromptFormData>({
    name: '',
    description: '',
    template: '',
    category: '',
    is_active: true,
    variables: []
  })
  const [testVariables, setTestVariables] = useState<Record<string, string>>({})
  const [validationResult, setValidationResult] = useState<string>('')

  // 加载数据
  const loadData = async () => {
    setLoading(true)
    try {
      const [promptList, categoryList] = await Promise.all([
        promptApi.getPrompts(selectedCategory, showActiveOnly),
        promptApi.getCategories()
      ])
      setPrompts(promptList)
      setCategories(categoryList)
    } catch (error) {
      console.error('Failed to load data:', error)
      setError('加载数据失败')
      toast.error('加载数据失败')
    } finally {
      setLoading(false)
    }
  }

  // 过滤后的Prompt列表
  const filteredPrompts = prompts.filter(prompt => {
    const matchesSearch = prompt.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         prompt.description.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesCategory = !selectedCategory || prompt.category === selectedCategory
    const matchesActive = !showActiveOnly || prompt.is_active
    
    return matchesSearch && matchesCategory && matchesActive
  })

  // 提取模板变量
  const extractVariables = (template: string): string[] => {
    const matches = template.match(/\{([^}]+)\}/g)
    if (!matches) return []
    return [...new Set(matches.map(match => match.slice(1, -1)))]
  }

  // 重置表单
  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      template: '',
      category: '',
      is_active: true,
      variables: []
    })
    setTestVariables({})
    setValidationResult('')
  }

  // 创建Prompt
  const handleCreate = async () => {
    try {
      const variables = extractVariables(formData.template)
      const promptData: PromptTemplateCreate = {
        ...formData,
        variables
      }
      
      const newPrompt = await promptApi.createPrompt(promptData)
      addPrompt(newPrompt)
      setShowCreateModal(false)
      resetForm()
      toast.success('Prompt模板创建成功')
    } catch (error) {
      console.error('Failed to create prompt:', error)
      toast.error('创建Prompt模板失败')
    }
  }

  // 更新Prompt
  const handleUpdate = async () => {
    if (!selectedPrompt) return
    
    try {
      const variables = extractVariables(formData.template)
      const updates: PromptTemplateUpdate = {
        ...formData,
        variables
      }
      
      const updatedPrompt = await promptApi.updatePrompt(selectedPrompt.id, updates)
      updatePrompt(selectedPrompt.id, updatedPrompt)
      setShowEditModal(false)
      resetForm()
      toast.success('Prompt模板更新成功')
    } catch (error) {
      console.error('Failed to update prompt:', error)
      toast.error('更新Prompt模板失败')
    }
  }

  // 删除Prompt
  const handleDelete = async (prompt: PromptTemplate) => {
    if (!window.confirm(`确定要删除Prompt模板 "${prompt.name}" 吗？`)) {
      return
    }
    
    try {
      await promptApi.deletePrompt(prompt.id)
      removePrompt(prompt.id)
      toast.success('Prompt模板删除成功')
    } catch (error) {
      console.error('Failed to delete prompt:', error)
      toast.error('删除Prompt模板失败')
    }
  }

  // 验证Prompt
  const handleValidate = async (prompt: PromptTemplate) => {
    try {
      const result = await promptApi.validatePrompt(prompt.id, testVariables)
      setValidationResult(result.message || '验证成功')
      toast.success('Prompt验证完成')
    } catch (error) {
      console.error('Failed to validate prompt:', error)
      setValidationResult('验证失败')
      toast.error('Prompt验证失败')
    }
  }

  // 复制Prompt
  const handleCopy = async (prompt: PromptTemplate) => {
    try {
      await navigator.clipboard.writeText(prompt.template)
      toast.success('Prompt模板已复制到剪贴板')
    } catch (error) {
      console.error('Failed to copy prompt:', error)
      toast.error('复制失败')
    }
  }

  // 打开编辑模态框
  const openEditModal = (prompt: PromptTemplate) => {
    setSelectedPrompt(prompt)
    setFormData({
      name: prompt.name,
      description: prompt.description,
      template: prompt.template,
      category: prompt.category,
      is_active: prompt.is_active,
      variables: prompt.variables
    })
    setShowEditModal(true)
  }

  // 打开查看模态框
  const openViewModal = (prompt: PromptTemplate) => {
    setSelectedPrompt(prompt)
    setShowViewModal(true)
  }

  // 组件挂载时加载数据
  useEffect(() => {
    loadData()
  }, [selectedCategory, showActiveOnly])

  // 模板变化时更新变量
  useEffect(() => {
    if (formData.template) {
      const variables = extractVariables(formData.template)
      setFormData(prev => ({ ...prev, variables }))
      
      // 初始化测试变量
      const newTestVariables: Record<string, string> = {}
      variables.forEach(variable => {
        if (!testVariables[variable]) {
          newTestVariables[variable] = ''
        }
      })
      setTestVariables(prev => ({ ...prev, ...newTestVariables }))
    }
  }, [formData.template])

  return (
    <div className="flex flex-col h-full bg-background">
      {/* 页面头部 */}
      <div className="flex-shrink-0 bg-card border-b border-border p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Prompt模板管理</h2>
            <p className="text-muted-foreground mt-1">
              创建和管理AI对话的Prompt模板
            </p>
          </div>
          <button
            onClick={() => {
              resetForm()
              setShowCreateModal(true)
            }}
            className="btn btn-primary"
          >
            <Plus className="h-4 w-4 mr-2" />
            新建模板
          </button>
        </div>

        {/* 搜索和过滤 */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex-1 min-w-[300px]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="搜索模板名称或描述..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input pl-10 w-full"
              />
            </div>
          </div>
          
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="select min-w-[150px]"
          >
            <option value="">所有分类</option>
            {categories.map(category => (
              <option key={category} value={category}>
                {category}
              </option>
            ))}
          </select>
          
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showActiveOnly}
              onChange={(e) => setShowActiveOnly(e.target.checked)}
              className="rounded border-input"
            />
            <span className="text-sm text-muted-foreground">仅显示激活的</span>
          </label>
          
          <button
            onClick={loadData}
            disabled={isLoading}
            className="btn btn-ghost"
            title="刷新数据"
          >
            <Filter className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* 模板列表 */}
      <div className="flex-1 overflow-y-auto p-6">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
              <p className="text-muted-foreground">加载中...</p>
            </div>
          </div>
        ) : filteredPrompts.length === 0 ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <p className="text-lg text-muted-foreground mb-2">暂无Prompt模板</p>
              <p className="text-sm text-muted-foreground mb-4">
                {searchTerm || selectedCategory ? '没有找到匹配的模板' : '开始创建您的第一个Prompt模板'}
              </p>
              <button
                onClick={() => {
                  resetForm()
                  setShowCreateModal(true)
                }}
                className="btn btn-primary"
              >
                <Plus className="h-4 w-4 mr-2" />
                创建模板
              </button>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredPrompts.map((prompt) => (
              <div key={prompt.id} className="card p-6 hover:shadow-md transition-shadow">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <h3 className="font-semibold text-foreground mb-1">
                      {prompt.name}
                    </h3>
                    <p className="text-sm text-muted-foreground line-clamp-2">
                      {prompt.description}
                    </p>
                  </div>
                  <div className="flex items-center space-x-1 ml-2">
                    <span className={`
                      badge text-xs
                      ${prompt.is_active ? 'badge-default' : 'badge-secondary'}
                    `}>
                      {prompt.is_active ? '激活' : '禁用'}
                    </span>
                  </div>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">分类:</span>
                    <span className="badge badge-outline">{prompt.category}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">变量:</span>
                    <span className="text-foreground">{prompt.variables.length} 个</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">更新:</span>
                    <span className="text-foreground">
                      {new Date(prompt.updated_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={() => openViewModal(prompt)}
                      className="btn btn-ghost p-2"
                      title="查看详情"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleCopy(prompt)}
                      className="btn btn-ghost p-2"
                      title="复制模板"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleValidate(prompt)}
                      className="btn btn-ghost p-2"
                      title="验证模板"
                    >
                      <TestTube className="h-4 w-4" />
                    </button>
                  </div>
                  <div className="flex items-center space-x-1">
                    <button
                      onClick={() => openEditModal(prompt)}
                      className="btn btn-ghost p-2"
                      title="编辑"
                    >
                      <Edit className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(prompt)}
                      className="btn btn-ghost p-2 text-destructive hover:text-destructive"
                      title="删除"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 创建模态框 */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-card rounded-lg shadow-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-foreground">创建Prompt模板</h3>
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="btn btn-ghost p-2"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    模板名称 *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    className="input w-full"
                    placeholder="输入模板名称"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    描述
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    className="textarea w-full"
                    rows={3}
                    placeholder="输入模板描述"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    分类
                  </label>
                  <input
                    type="text"
                    value={formData.category}
                    onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value }))}
                    className="input w-full"
                    placeholder="输入分类名称"
                    list="categories"
                  />
                  <datalist id="categories">
                    {categories.map(category => (
                      <option key={category} value={category} />
                    ))}
                  </datalist>
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    模板内容 *
                  </label>
                  <textarea
                    value={formData.template}
                    onChange={(e) => setFormData(prev => ({ ...prev, template: e.target.value }))}
                    className="textarea w-full font-mono"
                    rows={8}
                    placeholder="输入Prompt模板，使用 {变量名} 定义变量"
                  />
                  {formData.variables.length > 0 && (
                    <div className="mt-2">
                      <p className="text-sm text-muted-foreground mb-1">检测到的变量:</p>
                      <div className="flex flex-wrap gap-1">
                        {formData.variables.map(variable => (
                          <span key={variable} className="badge badge-outline text-xs">
                            {variable}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="is_active"
                    checked={formData.is_active}
                    onChange={(e) => setFormData(prev => ({ ...prev, is_active: e.target.checked }))}
                    className="rounded border-input"
                  />
                  <label htmlFor="is_active" className="text-sm text-foreground">
                    激活模板
                  </label>
                </div>
              </div>

              <div className="flex items-center justify-end space-x-3 mt-6 pt-6 border-t border-border">
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="btn btn-ghost"
                >
                  取消
                </button>
                <button
                  onClick={handleCreate}
                  disabled={!formData.name || !formData.template}
                  className="btn btn-primary"
                >
                  <Save className="h-4 w-4 mr-2" />
                  创建
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 编辑模态框 */}
      {showEditModal && selectedPrompt && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-card rounded-lg shadow-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-foreground">编辑Prompt模板</h3>
                <button
                  onClick={() => setShowEditModal(false)}
                  className="btn btn-ghost p-2"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    模板名称 *
                  </label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    className="input w-full"
                    placeholder="输入模板名称"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    描述
                  </label>
                  <textarea
                    value={formData.description}
                    onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    className="textarea w-full"
                    rows={3}
                    placeholder="输入模板描述"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    分类
                  </label>
                  <input
                    type="text"
                    value={formData.category}
                    onChange={(e) => setFormData(prev => ({ ...prev, category: e.target.value }))}
                    className="input w-full"
                    placeholder="输入分类名称"
                    list="categories"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-foreground mb-2">
                    模板内容 *
                  </label>
                  <textarea
                    value={formData.template}
                    onChange={(e) => setFormData(prev => ({ ...prev, template: e.target.value }))}
                    className="textarea w-full font-mono"
                    rows={8}
                    placeholder="输入Prompt模板，使用 {变量名} 定义变量"
                  />
                  {formData.variables.length > 0 && (
                    <div className="mt-2">
                      <p className="text-sm text-muted-foreground mb-1">检测到的变量:</p>
                      <div className="flex flex-wrap gap-1">
                        {formData.variables.map(variable => (
                          <span key={variable} className="badge badge-outline text-xs">
                            {variable}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="edit_is_active"
                    checked={formData.is_active}
                    onChange={(e) => setFormData(prev => ({ ...prev, is_active: e.target.checked }))}
                    className="rounded border-input"
                  />
                  <label htmlFor="edit_is_active" className="text-sm text-foreground">
                    激活模板
                  </label>
                </div>
              </div>

              <div className="flex items-center justify-end space-x-3 mt-6 pt-6 border-t border-border">
                <button
                  onClick={() => setShowEditModal(false)}
                  className="btn btn-ghost"
                >
                  取消
                </button>
                <button
                  onClick={handleUpdate}
                  disabled={!formData.name || !formData.template}
                  className="btn btn-primary"
                >
                  <Save className="h-4 w-4 mr-2" />
                  保存
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 查看模态框 */}
      {showViewModal && selectedPrompt && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-card rounded-lg shadow-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-foreground">
                  {selectedPrompt.name}
                </h3>
                <button
                  onClick={() => setShowViewModal(false)}
                  className="btn btn-ghost p-2"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>

              <div className="space-y-6">
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">描述</h4>
                  <p className="text-muted-foreground">
                    {selectedPrompt.description || '无描述'}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium text-foreground mb-2">分类</h4>
                    <span className="badge badge-outline">{selectedPrompt.category}</span>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-foreground mb-2">状态</h4>
                    <span className={`
                      badge
                      ${selectedPrompt.is_active ? 'badge-default' : 'badge-secondary'}
                    `}>
                      {selectedPrompt.is_active ? '激活' : '禁用'}
                    </span>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">模板内容</h4>
                  <div className="bg-muted p-4 rounded-lg">
                    <pre className="whitespace-pre-wrap text-sm text-foreground font-mono">
                      {selectedPrompt.template}
                    </pre>
                  </div>
                </div>

                {selectedPrompt.variables.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-foreground mb-2">变量列表</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedPrompt.variables.map(variable => (
                        <span key={variable} className="badge badge-outline">
                          {variable}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <h4 className="font-medium text-foreground mb-1">创建时间</h4>
                    <p className="text-muted-foreground">
                      {new Date(selectedPrompt.created_at).toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground mb-1">更新时间</h4>
                    <p className="text-muted-foreground">
                      {new Date(selectedPrompt.updated_at).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-end space-x-3 mt-6 pt-6 border-t border-border">
                <button
                  onClick={() => handleCopy(selectedPrompt)}
                  className="btn btn-ghost"
                >
                  <Copy className="h-4 w-4 mr-2" />
                  复制模板
                </button>
                <button
                  onClick={() => {
                    setShowViewModal(false)
                    openEditModal(selectedPrompt)
                  }}
                  className="btn btn-primary"
                >
                  <Edit className="h-4 w-4 mr-2" />
                  编辑
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default PromptsPage