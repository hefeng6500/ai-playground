import React, { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, RefreshCw, Trash2 } from 'lucide-react'
import { useChatStore, usePromptStore, generateId } from '../store'
import { chatApi, promptApi } from '../services/api'
import { toast } from 'sonner'

const ChatPage: React.FC = () => {
  const {
    messages,
    currentConversationId,
    isLoading,
    error,
    addMessage,
    setMessages,
    setLoading,
    setError,
    clearMessages,
    startNewConversation
  } = useChatStore()

  const { prompts, setPrompts } = usePromptStore()
  const [inputMessage, setInputMessage] = useState('')
  const [selectedPrompt, setSelectedPrompt] = useState<string>('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // 滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  // 加载Prompt模板
  const loadPrompts = async () => {
    try {
      const promptList = await promptApi.getPrompts()
      setPrompts(promptList)
    } catch (error) {
      console.error('Failed to load prompts:', error)
      toast.error('加载Prompt模板失败')
    }
  }

  // 发送消息
  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage = inputMessage.trim()
    setInputMessage('')
    setError(null)
    setLoading(true)

    // 添加用户消息
    addMessage({
      role: 'user',
      content: userMessage,
      conversationId: currentConversationId || undefined
    })

    try {
      // 发送API请求
      const response = await chatApi.sendMessage({
        message: userMessage,
        conversation_id: currentConversationId || undefined,
        use_prompt_template: selectedPrompt || undefined
      })

      // 添加AI回复
      addMessage({
        role: 'assistant',
        content: response.message,
        conversationId: response.conversation_id,
        promptTemplateUsed: response.prompt_template_used,
        metadata: response.metadata
      })

      // 更新当前会话ID
      if (!currentConversationId && response.conversation_id) {
        // 这里可以更新会话ID，但由于store设计，我们暂时不做处理
      }

    } catch (error) {
      console.error('Failed to send message:', error)
      setError('发送消息失败，请重试')
      toast.error('发送消息失败')
    } finally {
      setLoading(false)
    }
  }

  // 处理键盘事件
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // 清空聊天记录
  const handleClearChat = () => {
    if (window.confirm('确定要清空当前聊天记录吗？')) {
      clearMessages()
      startNewConversation()
      toast.success('聊天记录已清空')
    }
  }

  // 重新发送最后一条消息
  const handleRetry = () => {
    const lastUserMessage = messages.filter(m => m.role === 'user').pop()
    if (lastUserMessage) {
      setInputMessage(lastUserMessage.content)
      textareaRef.current?.focus()
    }
  }

  // 自动调整文本框高度
  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px'
    }
  }

  // 组件挂载时的初始化
  useEffect(() => {
    loadPrompts()
    if (!currentConversationId) {
      startNewConversation()
    }
  }, [])

  // 消息变化时滚动到底部
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // 输入框内容变化时调整高度
  useEffect(() => {
    adjustTextareaHeight()
  }, [inputMessage])

  return (
    <div className="flex flex-col h-full bg-background">
      {/* 聊天头部 */}
      <div className="flex-shrink-0 bg-card border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Bot className="h-6 w-6 text-primary" />
              <h3 className="text-lg font-semibold text-foreground">
                AI助手
              </h3>
            </div>
            
            {/* Prompt模板选择 */}
            <div className="flex items-center space-x-2">
              <label className="text-sm text-muted-foreground">模板:</label>
              <select
                value={selectedPrompt}
                onChange={(e) => setSelectedPrompt(e.target.value)}
                className="select text-sm min-w-[150px]"
              >
                <option value="">默认模板</option>
                {prompts.map((prompt) => (
                  <option key={prompt.id} value={prompt.id}>
                    {prompt.name}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleRetry}
              disabled={isLoading || messages.length === 0}
              className="btn btn-ghost p-2"
              title="重试最后一条消息"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
            <button
              onClick={handleClearChat}
              disabled={isLoading || messages.length === 0}
              className="btn btn-ghost p-2 text-destructive hover:text-destructive"
              title="清空聊天记录"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* 聊天消息区域 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="h-16 w-16 text-muted-foreground mb-4" />
            <h3 className="text-xl font-semibold text-foreground mb-2">
              欢迎使用LangChain助手
            </h3>
            <p className="text-muted-foreground max-w-md">
              我是您的AI助手，可以帮助您解答问题、处理任务。请输入您的问题开始对话。
            </p>
            {selectedPrompt && (
              <div className="mt-4 p-3 bg-muted rounded-lg">
                <p className="text-sm text-muted-foreground">
                  当前使用模板: {prompts.find(p => p.id === selectedPrompt)?.name}
                </p>
              </div>
            )}
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex items-start space-x-3 animate-fade-in ${
                  message.role === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                }`}
              >
                {/* 头像 */}
                <div className={`
                  flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                  ${message.role === 'user' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'bg-muted text-muted-foreground'
                  }
                `}>
                  {message.role === 'user' ? (
                    <User className="h-4 w-4" />
                  ) : (
                    <Bot className="h-4 w-4" />
                  )}
                </div>

                {/* 消息内容 */}
                <div className={`
                  flex-1 max-w-4xl
                  ${message.role === 'user' ? 'text-right' : 'text-left'}
                `}>
                  <div className={`
                    inline-block p-3 rounded-lg
                    ${message.role === 'user'
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-muted text-muted-foreground'
                    }
                  `}>
                    <div className="whitespace-pre-wrap break-words">
                      {message.content}
                    </div>
                    
                    {/* 消息元数据 */}
                    <div className="mt-2 text-xs opacity-70">
                      {new Date(message.timestamp).toLocaleTimeString()}
                      {message.promptTemplateUsed && (
                        <span className="ml-2">
                          · 模板: {prompts.find(p => p.id === message.promptTemplateUsed)?.name || message.promptTemplateUsed}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* 加载指示器 */}
            {isLoading && (
              <div className="flex items-start space-x-3 animate-fade-in">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted text-muted-foreground flex items-center justify-center">
                  <Bot className="h-4 w-4" />
                </div>
                <div className="flex-1">
                  <div className="inline-block p-3 rounded-lg bg-muted text-muted-foreground">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>AI正在思考中...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 错误提示 */}
            {error && (
              <div className="flex justify-center">
                <div className="bg-destructive/10 text-destructive px-4 py-2 rounded-lg text-sm">
                  {error}
                </div>
              </div>
            )}
          </>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="flex-shrink-0 bg-card border-t border-border p-4">
        <div className="flex items-end space-x-3">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="输入您的问题...（Shift+Enter换行，Enter发送）"
              className="textarea w-full resize-none"
              rows={1}
              disabled={isLoading}
            />
          </div>
          <button
            onClick={sendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="btn btn-primary p-3 flex-shrink-0"
            title="发送消息"
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
        
        {/* 输入提示 */}
        <div className="mt-2 text-xs text-muted-foreground text-center">
          {selectedPrompt ? (
            <span>
              使用模板: {prompts.find(p => p.id === selectedPrompt)?.name} · 
            </span>
          ) : null}
          <span>Shift+Enter 换行 · Enter 发送</span>
        </div>
      </div>
    </div>
  )
}

export default ChatPage