import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

// 聊天消息类型
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  conversationId?: string
  promptTemplateUsed?: string
  metadata?: Record<string, any>
}

// 会话历史类型
export interface ConversationHistory {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: string
  updatedAt: string
  metadata?: Record<string, any>
}

// Prompt模板类型
export interface PromptTemplate {
  id: string
  name: string
  description: string
  template: string
  variables: string[]
  category: string
  isActive: boolean
  createdAt: string
  updatedAt: string
  metadata?: Record<string, any>
}

// 系统配置类型
export interface SystemConfig {
  openaiModel: string
  maxTokens: number
  temperature: number
  defaultPromptTemplate?: string
  enableLogging: boolean
  logLevel: string
}

// 聊天状态
interface ChatState {
  messages: ChatMessage[]
  currentConversationId: string | null
  conversations: ConversationHistory[]
  isLoading: boolean
  error: string | null
  
  // Actions
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void
  setMessages: (messages: ChatMessage[]) => void
  setCurrentConversation: (conversationId: string | null) => void
  setConversations: (conversations: ConversationHistory[]) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  clearMessages: () => void
  startNewConversation: () => void
}

// Prompt模板状态
interface PromptState {
  prompts: PromptTemplate[]
  selectedPrompt: PromptTemplate | null
  isLoading: boolean
  error: string | null
  
  // Actions
  setPrompts: (prompts: PromptTemplate[]) => void
  setSelectedPrompt: (prompt: PromptTemplate | null) => void
  addPrompt: (prompt: PromptTemplate) => void
  updatePrompt: (id: string, updates: Partial<PromptTemplate>) => void
  removePrompt: (id: string) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

// 系统配置状态
interface ConfigState {
  config: SystemConfig | null
  isLoading: boolean
  error: string | null
  
  // Actions
  setConfig: (config: SystemConfig) => void
  updateConfig: (updates: Partial<SystemConfig>) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

// 应用状态
interface AppState {
  sidebarOpen: boolean
  currentPage: string
  theme: 'light' | 'dark'
  
  // Actions
  setSidebarOpen: (open: boolean) => void
  setCurrentPage: (page: string) => void
  setTheme: (theme: 'light' | 'dark') => void
  toggleSidebar: () => void
  toggleTheme: () => void
}

// 聊天状态管理
export const useChatStore = create<ChatState>()(devtools(
  (set, get) => ({
    messages: [],
    currentConversationId: null,
    conversations: [],
    isLoading: false,
    error: null,
    
    addMessage: (message) => {
      const newMessage: ChatMessage = {
        ...message,
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString()
      }
      
      set((state) => ({
        messages: [...state.messages, newMessage]
      }))
    },
    
    setMessages: (messages) => set({ messages }),
    
    setCurrentConversation: (conversationId) => set({ currentConversationId: conversationId }),
    
    setConversations: (conversations) => set({ conversations }),
    
    setLoading: (isLoading) => set({ isLoading }),
    
    setError: (error) => set({ error }),
    
    clearMessages: () => set({ messages: [] }),
    
    startNewConversation: () => {
      const conversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      set({
        currentConversationId: conversationId,
        messages: [],
        error: null
      })
    }
  }),
  { name: 'chat-store' }
))

// Prompt模板状态管理
export const usePromptStore = create<PromptState>()(devtools(
  (set, get) => ({
    prompts: [],
    selectedPrompt: null,
    isLoading: false,
    error: null,
    
    setPrompts: (prompts) => set({ prompts }),
    
    setSelectedPrompt: (prompt) => set({ selectedPrompt: prompt }),
    
    addPrompt: (prompt) => set((state) => ({
      prompts: [...state.prompts, prompt]
    })),
    
    updatePrompt: (id, updates) => set((state) => ({
      prompts: state.prompts.map(p => p.id === id ? { ...p, ...updates } : p),
      selectedPrompt: state.selectedPrompt?.id === id 
        ? { ...state.selectedPrompt, ...updates } 
        : state.selectedPrompt
    })),
    
    removePrompt: (id) => set((state) => ({
      prompts: state.prompts.filter(p => p.id !== id),
      selectedPrompt: state.selectedPrompt?.id === id ? null : state.selectedPrompt
    })),
    
    setLoading: (isLoading) => set({ isLoading }),
    
    setError: (error) => set({ error })
  }),
  { name: 'prompt-store' }
))

// 系统配置状态管理
export const useConfigStore = create<ConfigState>()(devtools(
  (set, get) => ({
    config: null,
    isLoading: false,
    error: null,
    
    setConfig: (config) => set({ config }),
    
    updateConfig: (updates) => set((state) => ({
      config: state.config ? { ...state.config, ...updates } : null
    })),
    
    setLoading: (isLoading) => set({ isLoading }),
    
    setError: (error) => set({ error })
  }),
  { name: 'config-store' }
))

// 应用状态管理
export const useAppStore = create<AppState>()(devtools(
  (set, get) => ({
    sidebarOpen: true,
    currentPage: 'chat',
    theme: 'light',
    
    setSidebarOpen: (open) => set({ sidebarOpen: open }),
    
    setCurrentPage: (page) => set({ currentPage: page }),
    
    setTheme: (theme) => set({ theme }),
    
    toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
    
    toggleTheme: () => set((state) => ({ 
      theme: state.theme === 'light' ? 'dark' : 'light' 
    }))
  }),
  { name: 'app-store' }
))

// 生成唯一ID的工具函数
export const generateId = (prefix: string = 'id') => {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

// 格式化时间的工具函数
export const formatTimestamp = (timestamp: string) => {
  const date = new Date(timestamp)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}