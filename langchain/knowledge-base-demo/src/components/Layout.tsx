import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useAppStore } from '../store'
import { 
  MessageSquare, 
  FileText, 
  Settings, 
  FileBarChart, 
  Menu, 
  X,
  Sun,
  Moon
} from 'lucide-react'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const location = useLocation()
  const { 
    sidebarOpen, 
    theme, 
    toggleSidebar, 
    toggleTheme, 
    setCurrentPage 
  } = useAppStore()

  const navigation = [
    {
      name: '聊天',
      href: '/chat',
      icon: MessageSquare,
      current: location.pathname === '/chat'
    },
    {
      name: 'Prompt管理',
      href: '/prompts',
      icon: FileText,
      current: location.pathname === '/prompts'
    },
    {
      name: '系统配置',
      href: '/config',
      icon: Settings,
      current: location.pathname === '/config'
    },
    {
      name: '日志查看',
      href: '/logs',
      icon: FileBarChart,
      current: location.pathname === '/logs'
    }
  ]

  React.useEffect(() => {
    const page = location.pathname.slice(1) || 'chat'
    setCurrentPage(page)
  }, [location.pathname, setCurrentPage])

  React.useEffect(() => {
    // 应用主题
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  return (
    <div className="flex h-screen bg-background">
      {/* 侧边栏 */}
      <div className={`
        ${sidebarOpen ? 'w-64' : 'w-0'} 
        transition-all duration-300 ease-in-out 
        bg-card border-r border-border 
        flex flex-col
        ${sidebarOpen ? 'opacity-100' : 'opacity-0'}
      `}>
        {sidebarOpen && (
          <>
            {/* 侧边栏头部 */}
            <div className="p-4 border-b border-border">
              <div className="flex items-center justify-between">
                <h1 className="text-lg font-semibold text-foreground">
                  LangChain助手
                </h1>
                <button
                  onClick={toggleSidebar}
                  className="p-1 rounded-md hover:bg-accent hover:text-accent-foreground"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>
              <p className="text-sm text-muted-foreground mt-1">
                智能对话与知识管理
              </p>
            </div>

            {/* 导航菜单 */}
            <nav className="flex-1 p-4 space-y-2">
              {navigation.map((item) => {
                const Icon = item.icon
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    className={`
                      flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors
                      ${item.current 
                        ? 'bg-primary text-primary-foreground' 
                        : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                      }
                    `}
                  >
                    <Icon className="mr-3 h-5 w-5" />
                    {item.name}
                  </Link>
                )
              })}
            </nav>

            {/* 侧边栏底部 */}
            <div className="p-4 border-t border-border">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  主题设置
                </span>
                <button
                  onClick={toggleTheme}
                  className="p-2 rounded-md hover:bg-accent hover:text-accent-foreground transition-colors"
                  title={theme === 'light' ? '切换到深色模式' : '切换到浅色模式'}
                >
                  {theme === 'light' ? (
                    <Moon className="h-4 w-4" />
                  ) : (
                    <Sun className="h-4 w-4" />
                  )}
                </button>
              </div>
              
              <div className="mt-3 text-xs text-muted-foreground">
                <div>版本: v1.0.0</div>
                <div>构建: {new Date().toLocaleDateString()}</div>
              </div>
            </div>
          </>
        )}
      </div>

      {/* 主内容区域 */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* 顶部导航栏 */}
        <header className="bg-card border-b border-border px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {!sidebarOpen && (
                <button
                  onClick={toggleSidebar}
                  className="p-2 rounded-md hover:bg-accent hover:text-accent-foreground transition-colors"
                  title="打开侧边栏"
                >
                  <Menu className="h-5 w-5" />
                </button>
              )}
              
              <div>
                <h2 className="text-lg font-semibold text-foreground">
                  {navigation.find(item => item.current)?.name || '首页'}
                </h2>
                <p className="text-sm text-muted-foreground">
                  {getPageDescription(location.pathname)}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              {/* 状态指示器 */}
              <div className="flex items-center space-x-2">
                <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" title="服务正常" />
                <span className="text-xs text-muted-foreground">在线</span>
              </div>
              
              {/* 主题切换按钮 */}
              <button
                onClick={toggleTheme}
                className="p-2 rounded-md hover:bg-accent hover:text-accent-foreground transition-colors"
                title={theme === 'light' ? '切换到深色模式' : '切换到浅色模式'}
              >
                {theme === 'light' ? (
                  <Moon className="h-4 w-4" />
                ) : (
                  <Sun className="h-4 w-4" />
                )}
              </button>
            </div>
          </div>
        </header>

        {/* 页面内容 */}
        <main className="flex-1 overflow-hidden">
          {children}
        </main>
      </div>

      {/* 移动端侧边栏遮罩 */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={toggleSidebar}
        />
      )}
    </div>
  )
}

// 获取页面描述
function getPageDescription(pathname: string): string {
  switch (pathname) {
    case '/chat':
      return '与AI助手进行智能对话'
    case '/prompts':
      return '管理和编辑Prompt模板'
    case '/config':
      return '配置系统参数和模型设置'
    case '/logs':
      return '查看系统运行日志和统计'
    default:
      return '欢迎使用LangChain知识库助手'
  }
}

export default Layout