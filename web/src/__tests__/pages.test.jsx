import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from '../App.jsx';
import { AuthProvider } from '../auth.jsx';
import patientScan from './fixtures/patient-scan.json';
import patientList from './fixtures/patient-list.json';
import clinicianScan from './fixtures/clinician-scan.json';
import clinicianList from './fixtures/clinician-list.json';

const HEALTH = { server: 'ok', ml: { status: 'ready', ready: true, device: 'cpu', networks: 20 }, agent: { configured: true, model: 'test-model' }, clinicianSignup: true, clinicianInviteRequired: true };
const CARD = {
  version: 'v4', lib_version: '4.0', created: '2026-10-19', profile: 'full', limitations: ['Not validated on phone photos.'],
  backbones: { convnext_tiny: { resolution: 224 }, swin_t: { resolution: 224 } }, folds_trained: 5, networks_loaded: 10, tta_views: 2, device: 'cpu',
  data: { n_lesions: 401059, n_patients: 1042, n_malignant: 393 },
  image_only_calibration: { source: 'oof', n_lesions: 401059, metrics: { auroc: 0.93, pauc_tpr80: 0.14 }, operating_points: [{ rule: 'sens95', label: 'keeps 95% of cancers', threshold: 0.61, specificity: 0.62 }] },
  results: { image_only: [{ model: 'image mean', AUROC: 0.931, pAUC80: 0.141 }] },
};

function mockFetch(user, scanFixture, listFixture, calls = []) {
  vi.stubGlobal('fetch', vi.fn(async (url, opts = {}) => {
    const path = String(url).replace(/^\/api/, '');
    const method = opts.method || 'GET';
    calls.push(`${method} ${path}`);
    const ok = (d, status = 200) => ({ ok: status < 400, status, json: async () => d });
    if (path === '/auth/me') return user ? ok({ user }) : ok({ message: 'no' }, 401);
    if (path === '/health') return ok(HEALTH);
    if (path === '/model-card') return ok(CARD);
    if (path.startsWith('/scans/') && method === 'GET') return ok(scanFixture);
    if (path.startsWith('/scans') && method === 'GET') return ok(listFixture);
    return ok({ message: `unmocked ${method} ${path}` }, 404);
  }));
}

const patient = { id: 1, name: 'Asha Patil', email: 'asha@example.com', role: 'patient' };
const clinician = { id: 2, name: 'Dr Rao', email: 'rao@clinic.org', role: 'clinician', organisation: 'City Skin Clinic' };

function renderAt(path) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AuthProvider><App /></AuthProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  HTMLDialogElement.prototype.showModal ??= function showModal() { this.open = true; };
  HTMLDialogElement.prototype.close ??= function close() { this.open = false; };
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('sign-in page', () => {
  it('shows the sign-in form, and the role choice plus invite code when registering', async () => {
    mockFetch(null);
    renderAt('/login');
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeTruthy();
    fireEvent.click(screen.getByText('Create an account'));
    expect(await screen.findByRole('heading', { name: 'Create your account' })).toBeTruthy();
    const radios = screen.getAllByRole('radio');
    expect(radios.map((r) => r.value)).toEqual(['patient', 'clinician']);
    expect(screen.queryByLabelText('Clinician invite code')).toBeNull();
    fireEvent.click(radios[1]);
    expect(await screen.findByLabelText('Clinician invite code')).toBeTruthy();
  });

  it('redirects signed-out visitors from a protected page to sign-in', async () => {
    mockFetch(null);
    renderAt('/history');
    expect(await screen.findByRole('heading', { name: 'Sign in' })).toBeTruthy();
  });
});

describe('patient pages', () => {
  it('renders the result in plain language, without clinician-only numbers', async () => {
    const calls = [];
    mockFetch(patient, patientScan, patientList, calls);
    renderAt(`/scan/${patientScan.scan.id}`);
    expect(await screen.findByRole('heading', { name: 'Elevated concern', level: 1 })).toBeTruthy();
    expect(screen.getByText(/Suggested timing: Within 2 to 4 weeks/)).toBeTruthy();
    for (const t of ['Grad-CAM', 'LIME', 'SHAP']) expect(screen.getByRole('heading', { name: t })).toBeTruthy();
    expect(screen.getAllByRole('img').some((i) => (i.getAttribute('src') || '').includes('/files/gradcam'))).toBe(true);
    expect(await screen.findByText('Book a skin check with a dermatologist.')).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'See a doctor sooner if' })).toBeTruthy();
    expect(screen.queryByRole('heading', { name: 'Model output' })).toBeNull();
    expect(screen.queryByText(/Type if it is a skin cancer/)).toBeNull();
    expect(document.body.textContent).not.toMatch(/z = 2\.31/);
    expect(screen.getByRole('link', { name: /Download report/ }).getAttribute('href')).toBe(`/api/scans/${patientScan.scan.id}/report.pdf`);
    expect(calls.filter((c) => c.startsWith('POST'))).toEqual([]);
  });

  it('lists past scans on the home page', async () => {
    mockFetch(patient, patientScan, patientList);
    renderAt('/');
    expect(await screen.findByRole('heading', { name: /Hello, Asha/ })).toBeTruthy();
    expect(await screen.findAllByText('Elevated concern')).not.toHaveLength(0);
    expect(screen.getAllByRole('link', { name: 'Check a spot' }).length).toBe(2);
  });

  it('shows the history page with a search box', async () => {
    mockFetch(patient, patientScan, patientList);
    renderAt('/history');
    expect(await screen.findByRole('heading', { name: 'History' })).toBeTruthy();
    expect(await screen.findByPlaceholderText(/Search by name/)).toBeTruthy();
  });

  it('shows the new-scan form with the photo chooser and symptom chips', async () => {
    mockFetch(patient);
    renderAt('/scan/new');
    expect(await screen.findByRole('heading', { name: 'Check a spot' })).toBeTruthy();
    expect(screen.getByLabelText('Itching')).toBeTruthy();
    expect(screen.queryByLabelText('Patient reference')).toBeNull();
    expect(screen.getByRole('button', { name: /Analyse this spot/ }).disabled).toBe(true);
  });

  it('keeps the model card to clinicians but explains the tool to patients', async () => {
    mockFetch(patient);
    renderAt('/about');
    expect(await screen.findByRole('heading', { name: 'About this tool' })).toBeTruthy();
    expect(await screen.findByText('Not validated on phone photos.')).toBeTruthy();
    expect(screen.queryByRole('heading', { name: 'Model card' })).toBeNull();
  });
});

describe('clinician pages', () => {
  it('renders scores, evidence, differential, options and the disposition rule', async () => {
    mockFetch(clinician, clinicianScan, clinicianList);
    renderAt(`/scan/${clinicianScan.scan.id}`);
    expect(await screen.findByRole('heading', { name: 'High concern', level: 1 })).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Model output' })).toBeTruthy();
    expect(screen.getByText('Type if it is a skin cancer')).toBeTruthy();
    expect(screen.getByText('z = 2.31')).toBeTruthy();
    expect(screen.getAllByRole('table').length).toBeGreaterThanOrEqual(2);
    expect(await screen.findByRole('heading', { name: 'Differential considerations' })).toBeTruthy();
    expect(screen.getByText('Wide local excision.')).toBeTruthy();
    expect(screen.getByText('Imiquimod 5% cream')).toBeTruthy();
    expect(screen.getAllByText(/Disposition rule for this tier/).length).toBeGreaterThan(0);
    expect(screen.getByText(/Agreement between the three maps/)).toBeTruthy();
    for (const t of ['Grad-CAM', 'LIME', 'SHAP']) expect(screen.getByRole('heading', { name: t })).toBeTruthy();
  });

  it('asks for a patient reference on the new-analysis form', async () => {
    mockFetch(clinician);
    renderAt('/scan/new');
    expect(await screen.findByRole('heading', { name: 'New analysis' })).toBeTruthy();
    expect(screen.getByLabelText('Patient reference')).toBeTruthy();
  });

  it('shows the model card with operating points', async () => {
    mockFetch(clinician);
    renderAt('/about');
    expect(await screen.findByRole('heading', { name: 'Model card' })).toBeTruthy();
    expect(await screen.findByText('keeps 95% of cancers')).toBeTruthy();
    expect(screen.getByText(/AUROC 0\.9300/)).toBeTruthy();
  });
});
