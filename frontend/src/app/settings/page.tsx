'use client';

import { useEffect } from 'react';
import { useSettingsStore } from '@/store/settingsStore';

export default function SettingsPage() {
  const { settings, isLoading, loadSettings, updateSettings } = useSettingsStore();

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // In a real app, you'd send this to the backend
    alert('Settings updated (frontend only for now)');
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="embedModel" className="block text-sm font-medium text-gray-700">
            Embedding Model
          </label>
          <input
            type="text"
            id="embedModel"
            value={settings.embedModel ?? ''}
            onChange={(e) => updateSettings({ embedModel: e.target.value })}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="rerankerModel" className="block text-sm font-medium text-gray-700">
            Reranker Model
          </label>
          <input
            type="text"
            id="rerankerModel"
            value={settings.rerankerModel ?? ''}
            onChange={(e) => updateSettings({ rerankerModel: e.target.value })}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="llmModel" className="block text-sm font-medium text-gray-700">
            LLM Model
          </label>
          <input
            type="text"
            id="llmModel"
            value={settings.llmModel ?? ''}
            onChange={(e) => updateSettings({ llmModel: e.target.value })}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="dataDir" className="block text-sm font-medium text-gray-700">
            Data Directory
          </label>
          <input
            type="text"
            id="dataDir"
            value={settings.dataDir ?? ''}
            onChange={(e) => updateSettings({ dataDir: e.target.value })}
            className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <button
            type="submit"
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Save Settings
          </button>
        </div>
      </form>
    </div>
  );
}