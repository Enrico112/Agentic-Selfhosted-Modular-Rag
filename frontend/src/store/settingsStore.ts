import { create } from 'zustand';

export interface Settings {
  embedModel: string;
  rerankerModel: string;
  llmModel: string;
  dataDir: string;
}

export interface SettingsState {
  settings: Settings;
  isLoading: boolean;
  updateSettings: (settings: Partial<Settings>) => void;
  loadSettings: () => Promise<void>;
  setLoading: (loading: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set, get) => ({
  settings: {
    embedModel: '',
    rerankerModel: '',
    llmModel: '',
    dataDir: '',
  },
  isLoading: false,
  updateSettings: (newSettings) => set((state) => ({
    settings: { ...state.settings, ...newSettings }
  })),
  loadSettings: async () => {
    set({ isLoading: true });
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/settings`);
      const data = await response.json();
      set({ settings: data });
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      set({ isLoading: false });
    }
  },
  setLoading: (loading) => set({ isLoading: loading }),
}));