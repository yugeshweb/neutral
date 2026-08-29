import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { EhrValidationView } from './components/EhrValidationView'
import './index.css'
import App from './App.tsx'

const validationSurface = window.location.pathname === '/ehr-validation'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {validationSurface ? <EhrValidationView /> : <App />}
  </StrictMode>,
)
