import React from 'react'
import { Outlet, NavLink } from 'react-router-dom'
import { Activity, Shield, TrendingUp, MessageSquare, Settings } from 'lucide-react'

const Layout = () => {
  const navItems = [
    { to: '/', label: 'Dashboard', icon: Activity },
    { to: '/pwsa', label: 'PWSA', icon: Shield },
    { to: '/finance', label: 'Finance', icon: TrendingUp },
    { to: '/llm', label: 'LLM', icon: MessageSquare },
    { to: '/settings', label: 'Settings', icon: Settings },
  ]

  return (
    <div className="min-h-screen bg-slate-900">
      {/* Navigation */}
      <nav className="bg-slate-800 border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold text-primary-400">PRISM-AI</h1>
              </div>
              <div className="hidden md:block">
                <div className="ml-10 flex items-baseline space-x-4">
                  {navItems.map((item) => {
                    const Icon = item.icon
                    return (
                      <NavLink
                        key={item.to}
                        to={item.to}
                        end={item.to === '/'}
                        className={({ isActive }) =>
                          `px-3 py-2 rounded-md text-sm font-medium flex items-center gap-2 transition-colors ${
                            isActive
                              ? 'bg-primary-600 text-white'
                              : 'text-slate-300 hover:bg-slate-700 hover:text-white'
                          }`
                        }
                      >
                        <Icon size={18} />
                        {item.label}
                      </NavLink>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout
