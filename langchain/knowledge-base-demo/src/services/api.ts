import axios, { AxiosResponse } from "axios";
import { toast } from "sonner";

// API基础配置
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8080";

// 创建axios实例
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error("API Request Error:", error);
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response: AxiosResponse) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error("API Response Error:", error);

    // 统一错误处理
    const message = error.response?.data?.detail || error.message || "请求失败";
    toast.error(`API错误: ${message}`);

    return Promise.reject(error);
  }
);

// 类型定义
export interface ChatRequest {
  message: string;
  conversation_id?: string;
  use_prompt_template?: string;
  context?: Record<string, any>;
}

export interface ChatResponse {
  message: string;
  conversation_id: string;
  timestamp: string;
  prompt_template_used?: string;
  metadata?: Record<string, any>;
}

export interface PromptTemplate {
  id: string;
  name: string;
  description: string;
  template: string;
  variables: string[];
  category: string;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

export interface PromptTemplateCreate {
  name: string;
  description: string;
  template: string;
  variables: string[];
  category: string;
  is_active?: boolean;
  metadata?: Record<string, any>;
}

export interface PromptTemplateUpdate {
  name?: string;
  description?: string;
  template?: string;
  variables?: string[];
  category?: string;
  is_active?: boolean;
  metadata?: Record<string, any>;
}

export interface SystemConfig {
  openai_model: string;
  max_tokens: number;
  temperature: number;
  default_prompt_template?: string;
  enable_logging: boolean;
  log_level: string;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  level: string;
  source: string;
  message: string;
  metadata?: Record<string, any>;
}

export interface ConversationHistory {
  id: string;
  title: string;
  messages: Array<{
    role: string;
    content: string;
    timestamp: string;
  }>;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  error?: string;
}

// 聊天API
export const chatApi = {
  // 发送消息
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>("/api/chat/send", request);
    return response.data;
  },

  // 获取会话列表
  getConversations: async (
    limit: number = 20
  ): Promise<ConversationHistory[]> => {
    const response = await api.get<ConversationHistory[]>(
      `/api/chat/conversations?limit=${limit}`
    );
    return response.data;
  },

  // 获取特定会话
  getConversation: async (
    conversationId: string
  ): Promise<ConversationHistory> => {
    const response = await api.get<ConversationHistory>(
      `/api/chat/conversations/${conversationId}`
    );
    return response.data;
  },

  // 删除会话
  deleteConversation: async (conversationId: string): Promise<ApiResponse> => {
    const response = await api.delete<ApiResponse>(
      `/api/chat/conversations/${conversationId}`
    );
    return response.data;
  },

  // 测试聊天功能
  testChat: async (): Promise<ApiResponse> => {
    const response = await api.post<ApiResponse>("/api/chat/test");
    return response.data;
  },
};

// Prompt模板API
export const promptApi = {
  // 获取所有模板
  getPrompts: async (
    category?: string,
    activeOnly: boolean = true
  ): Promise<PromptTemplate[]> => {
    const params = new URLSearchParams();
    if (category) params.append("category", category);
    if (activeOnly) params.append("active_only", "true");

    const response = await api.get<PromptTemplate[]>(
      `/api/prompts?${params.toString()}`
    );
    return response.data;
  },

  // 获取特定模板
  getPrompt: async (promptId: string): Promise<PromptTemplate> => {
    const response = await api.get<PromptTemplate>(`/api/prompts/${promptId}`);
    return response.data;
  },

  // 创建模板
  createPrompt: async (
    prompt: PromptTemplateCreate
  ): Promise<PromptTemplate> => {
    const response = await api.post<PromptTemplate>("/api/prompts", prompt);
    return response.data;
  },

  // 更新模板
  updatePrompt: async (
    promptId: string,
    updates: PromptTemplateUpdate
  ): Promise<PromptTemplate> => {
    const response = await api.put<PromptTemplate>(
      `/api/prompts/${promptId}`,
      updates
    );
    return response.data;
  },

  // 删除模板
  deletePrompt: async (promptId: string): Promise<ApiResponse> => {
    const response = await api.delete<ApiResponse>(`/api/prompts/${promptId}`);
    return response.data;
  },

  // 验证模板
  validatePrompt: async (
    promptId: string,
    testVariables?: Record<string, any>
  ): Promise<ApiResponse> => {
    const response = await api.post<ApiResponse>(
      `/api/prompts/${promptId}/validate`,
      { test_variables: testVariables }
    );
    return response.data;
  },

  // 获取分类列表
  getCategories: async (): Promise<string[]> => {
    const response = await api.get<string[]>("/api/prompts/categories/list");
    return response.data;
  },
};

// 系统配置API
export const configApi = {
  // 获取配置
  getConfig: async (): Promise<SystemConfig> => {
    const response = await api.get<SystemConfig>("/api/config");
    return response.data;
  },

  // 更新配置
  updateConfig: async (
    updates: Partial<SystemConfig>
  ): Promise<SystemConfig> => {
    const response = await api.put<SystemConfig>("/api/config", updates);
    return response.data;
  },

  // 重置配置
  resetConfig: async (): Promise<ApiResponse> => {
    const response = await api.post<ApiResponse>("/api/config/reset");
    return response.data;
  },

  // 获取可用模型
  getAvailableModels: async (): Promise<
    Array<{
      id: string;
      name: string;
      description: string;
      max_tokens: number;
    }>
  > => {
    const response = await api.get("/api/config/models");
    return response.data;
  },

  // 测试配置
  testConfig: async (): Promise<ApiResponse> => {
    const response = await api.post<ApiResponse>("/api/config/test");
    return response.data;
  },
};

// 日志API
export const logApi = {
  // 获取日志
  getLogs: async (
    params: {
      level?: string;
      start_time?: string;
      end_time?: string;
      limit?: number;
      offset?: number;
      search?: string;
    } = {}
  ): Promise<LogEntry[]> => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, value.toString());
      }
    });

    const response = await api.get<LogEntry[]>(
      `/api/logs?${searchParams.toString()}`
    );
    return response.data;
  },

  // 获取日志统计
  getLogStats: async (
    hours: number = 24
  ): Promise<{
    time_range: {
      start_time: string;
      end_time: string;
      hours: number;
    };
    total_logs: number;
    level_distribution: Record<string, number>;
    recent_errors: Array<{
      timestamp: string;
      level: string;
      message: string;
      source: string;
    }>;
    error_rate: number;
  }> => {
    const response = await api.get(`/api/logs/stats?hours=${hours}`);
    return response.data;
  },

  // 清理日志
  clearLogs: async (params: {
    before_date?: string;
    level?: string;
    confirm: boolean;
  }): Promise<ApiResponse> => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, value.toString());
      }
    });

    const response = await api.delete<ApiResponse>(
      `/api/logs?${searchParams.toString()}`
    );
    return response.data;
  },

  // 导出日志
  exportLogs: async (
    params: {
      format?: "json" | "csv";
      level?: string;
      start_time?: string;
      end_time?: string;
    } = {}
  ): Promise<ApiResponse> => {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.append(key, value.toString());
      }
    });

    const response = await api.get<ApiResponse>(
      `/api/logs/export?${searchParams.toString()}`
    );
    return response.data;
  },
};

// 健康检查API
export const healthApi = {
  // 检查API健康状态
  checkHealth: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await api.get("/api/health");
    return response.data;
  },

  // 检查根路径
  checkRoot: async (): Promise<{
    message: string;
    version: string;
    timestamp: string;
  }> => {
    const response = await api.get("/");
    return response.data;
  },
};

export default api;
