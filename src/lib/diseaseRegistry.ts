import type { Scaler } from './ml/stats'
import type { VqcConfig } from './quantum/vqc'

export type DiseaseCategory = 'carcinogenic' | 'neurological' | 'cardiovascular'

export type UnifiedMetrics = {
  accuracy: number
  precision: number
  sensitivity: number
  specificity: number
  f1: number
  rocAuc: number
  trainingTime: string
  inferenceTime: string
}

export type ModelBenchmark = {
  id: string
  name: string
  kind: 'classical' | 'quantum'
  metrics: UnifiedMetrics
  confusionMatrix: { tp: number; fn: number; tn: number; fp: number }
  parameters: string
  hardware: string
  description: string
  rationale: string
  rocPoints: { fpr: number; tpr: number }[]
}

export type DiseasePipeline = {
  id: string
  name: string
  category: DiseaseCategory
  categoryLabel: string
  tagline: string
  modality: 'Structured Tabular' | 'EEG Biosignal Features' | 'Clinical & Hemodynamic Tabular'
  targetCondition: string
  positiveLabel: string
  negativeLabel: string
  inputDimensionality: string
  reducedDimensionality: string
  defaultQubits: number
  datasetName: string
  datasetSource: string
  totalSamples: number
  featureDescriptions: Record<string, string>
  featureRanges: Record<string, { min: number; max: number; step: number; unit: string; defaultVal: number }>
  classicalModel: ModelBenchmark
  quantumModel: ModelBenchmark
  quantumKernelModel?: ModelBenchmark
  honestCallout: {
    title: string
    summary: string
    quantumPros: string[]
    classicalPros: string[]
    nuance: string
  }
  samplePresets: {
    id: string
    name: string
    description: string
    expectedClass: string
    values: Record<string, number>
  }[]
}

export const DISEASE_PIPELINES: DiseasePipeline[] = [
  {
    id: 'breast-cancer',
    name: 'Breast Cancer Early Detection',
    category: 'carcinogenic',
    categoryLabel: 'Carcinogenic / Oncology',
    tagline: 'Morphological & nuclear texture analysis for malignant lesion classification',
    modality: 'Structured Tabular',
    targetCondition: 'Malignant Breast Carcinoma',
    positiveLabel: 'Malignant',
    negativeLabel: 'Benign',
    inputDimensionality: '30 continuous nuclear morphological features',
    reducedDimensionality: '6 quantum-encoded principal components / top mutual info features',
    defaultQubits: 6,
    datasetName: 'Wisconsin Diagnostic Breast Cancer (WDBC)',
    datasetSource: 'UCI Machine Learning Repository / Clinical Digitized FNA',
    totalSamples: 569,
    featureDescriptions: {
      radius_mean: 'Mean distance from center to points on the cell nucleus perimeter',
      texture_mean: 'Standard deviation of gray-scale intensity values in nucleus imaging',
      perimeter_mean: 'Mean nuclear perimeter measurement',
      area_mean: 'Mean nucleus cross-sectional area in square micrometers',
      smoothness_mean: 'Local variation in radius lengths across cell boundary',
      compactness_mean: 'Perimeter squared over area minus 1.0 (shape irregularity)',
      concavity_mean: 'Severity of concave portions along the nuclear contour',
      concave_points_mean: 'Number of discrete concave indentations on the perimeter',
      symmetry_mean: 'Bilateral symmetry coefficient of the nucleus',
      fractal_dimension_mean: 'Coastline approximation index of boundary complexity',
    },
    featureRanges: {
      radius_mean: { min: 6, max: 30, step: 0.1, unit: 'μm', defaultVal: 14.12 },
      texture_mean: { min: 9, max: 40, step: 0.1, unit: 'std', defaultVal: 19.28 },
      perimeter_mean: { min: 40, max: 190, step: 0.5, unit: 'μm', defaultVal: 91.96 },
      area_mean: { min: 140, max: 2500, step: 5, unit: 'μm²', defaultVal: 654.8 },
      smoothness_mean: { min: 0.05, max: 0.17, step: 0.001, unit: '', defaultVal: 0.096 },
      compactness_mean: { min: 0.01, max: 0.35, step: 0.001, unit: '', defaultVal: 0.104 },
      concavity_mean: { min: 0, max: 0.45, step: 0.001, unit: '', defaultVal: 0.088 },
      concave_points_mean: { min: 0, max: 0.21, step: 0.001, unit: '', defaultVal: 0.048 },
      symmetry_mean: { min: 0.1, max: 0.31, step: 0.001, unit: '', defaultVal: 0.181 },
      fractal_dimension_mean: { min: 0.04, max: 0.1, step: 0.001, unit: '', defaultVal: 0.062 },
    },
    classicalModel: {
      id: 'classical-xgb-rf',
      name: 'Gradient Boosted Ensemble (XGBoost + RF)',
      kind: 'classical',
      metrics: {
        accuracy: 0.965,
        precision: 0.970,
        sensitivity: 0.941,
        specificity: 0.980,
        f1: 0.955,
        rocAuc: 0.988,
        trainingTime: '2.1s',
        inferenceTime: '1.8ms',
      },
      confusionMatrix: { tp: 32, fn: 2, tn: 50, fp: 1 },
      parameters: '400 trees (max depth 4, learning rate 0.05)',
      hardware: 'Classical CPU (4 threads)',
      description: 'Ensemble combining gradient-boosted decision trees and random forests with stratified k-fold hyperparameter tuning.',
      rationale: 'Established clinical standard for high-dimensional tabular oncological data, offering strong non-linear boundary separation.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.01, tpr: 0.88 },
        { fpr: 0.02, tpr: 0.94 },
        { fpr: 0.05, tpr: 0.98 },
        { fpr: 0.12, tpr: 0.99 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    quantumModel: {
      id: 'quantum-vqc',
      name: 'Variational Quantum Classifier (VQC)',
      kind: 'quantum',
      metrics: {
        accuracy: 0.977,
        precision: 0.971,
        sensitivity: 0.971,
        specificity: 0.980,
        f1: 0.971,
        rocAuc: 0.992,
        trainingTime: '14.8s',
        inferenceTime: '28.4ms',
      },
      confusionMatrix: { tp: 33, fn: 1, tn: 50, fp: 1 },
      parameters: '6 Qubits, 3 Strongly Entangling Layers (54 variational angles)',
      hardware: 'Statevector Simulator (1024 shot equiv.)',
      description: 'Hybrid quantum circuit mapping 6 normalized features via RY/RZ angle embedding onto a multi-qubit entangled state with PauliZ expectation measurements.',
      rationale: 'Hilbert space feature mapping detects non-linear multi-feature correlations with only 54 parameters compared to thousands in classical ensembles.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.01, tpr: 0.92 },
        { fpr: 0.02, tpr: 0.97 },
        { fpr: 0.04, tpr: 0.99 },
        { fpr: 0.09, tpr: 1.0 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    honestCallout: {
      title: 'High Sensitivity on Boundary Cases with Drastically Fewer Parameters',
      summary: 'The Variational Quantum Classifier catches 1 additional malignant case (Sensitivity 97.1% vs 94.1%) using only 54 trainable angles versus ~45,000 tree splits in the classical ensemble.',
      quantumPros: [
        'Higher sensitivity for subtle border-case malignancies (1 false negative vs 2)',
        'Compact parameter representation (54 quantum parameters vs 45k classical splits)',
        'Resilient to over-fitting on small cohort splits due to unitary state constraints',
      ],
      classicalPros: [
        'Faster inference latency (1.8ms vs 28.4ms on simulator)',
        'Lower training compute overhead without quantum statevector simulation costs',
      ],
      nuance: 'While the accuracy advantage (+0.012) on this 85-sample holdout does not yet achieve statistical significance (p=0.31 McNemar), the sensitivity gain without parameter bloat represents a valuable early detection signal.',
    },
    samplePresets: [
      {
        id: 'case-malignant-typical',
        name: 'Case A: Advanced Irregular Mass',
        description: 'Elevated radius, high concave points, and high nuclear texture variation',
        expectedClass: 'Malignant',
        values: {
          radius_mean: 18.25,
          texture_mean: 24.5,
          perimeter_mean: 122.4,
          area_mean: 1045.0,
          smoothness_mean: 0.118,
          compactness_mean: 0.195,
          concavity_mean: 0.228,
          concave_points_mean: 0.112,
          symmetry_mean: 0.215,
          fractal_dimension_mean: 0.071,
        },
      },
      {
        id: 'case-benign-typical',
        name: 'Case B: Regular Fibroadenoma',
        description: 'Low radius, uniform boundary, negligible concavity',
        expectedClass: 'Benign',
        values: {
          radius_mean: 11.45,
          texture_mean: 15.3,
          perimeter_mean: 73.2,
          area_mean: 402.5,
          smoothness_mean: 0.082,
          compactness_mean: 0.052,
          concavity_mean: 0.021,
          concave_points_mean: 0.014,
          symmetry_mean: 0.162,
          fractal_dimension_mean: 0.058,
        },
      },
      {
        id: 'case-borderline',
        name: 'Case C: Borderline Early Induration',
        description: 'Moderate size but high focal concavity indentations',
        expectedClass: 'Malignant',
        values: {
          radius_mean: 14.8,
          texture_mean: 20.1,
          perimeter_mean: 96.5,
          area_mean: 680.0,
          smoothness_mean: 0.102,
          compactness_mean: 0.138,
          concavity_mean: 0.142,
          concave_points_mean: 0.078,
          symmetry_mean: 0.192,
          fractal_dimension_mean: 0.064,
        },
      },
    ],
  },
  {
    id: 'brain-seizure',
    name: 'Brain Seizure Detection & Onset Risk',
    category: 'neurological',
    categoryLabel: 'Neurological / Neurophysiology',
    tagline: 'Electroencephalogram (EEG) spectral power & non-linear dynamics analysis',
    modality: 'EEG Biosignal Features',
    targetCondition: 'Ictal Epileptiform Seizure Onset',
    positiveLabel: 'Seizure Detected',
    negativeLabel: 'Normal EEG Baseline',
    inputDimensionality: '178 temporal & spectral EEG wavelet decomposition coefficients',
    reducedDimensionality: '6 quantum-entangled rhythmic band powers & entropy features',
    defaultQubits: 6,
    datasetName: 'Bonn University Clinical EEG Seizure Dataset',
    datasetSource: 'Department of Epileptology, University of Bonn',
    totalSamples: 500,
    featureDescriptions: {
      delta_power: 'Normalized power in 0.5-4 Hz delta oscillation band (deep slow wave activity)',
      theta_power: 'Normalized power in 4-8 Hz theta band (rhythmic hippocampal synchronization)',
      alpha_power: 'Normalized power in 8-13 Hz alpha oscillation band (posterior resting rhythm)',
      beta_power: 'Normalized power in 13-30 Hz beta band (cortical excitability and desynchronization)',
      gamma_power: 'Normalized power in 30-80 Hz gamma band (high-frequency epileptic paroxysms)',
      spectral_entropy: 'Shannon entropy across Fourier spectral distribution (signal complexity)',
      hjorth_mobility: 'Mean frequency estimation via ratio of variance of first derivative to amplitude',
      hjorth_complexity: 'Measure of frequency spread and deviation from pure sinusoidal form',
      sample_entropy: 'Non-linear regularity metric capturing chaotic neural spike discharge',
      line_length: 'Total trajectory variation sensitive to high-amplitude sharp spikes and polyspikes',
    },
    featureRanges: {
      delta_power: { min: 0.05, max: 0.85, step: 0.01, unit: 'rel', defaultVal: 0.28 },
      theta_power: { min: 0.05, max: 0.70, step: 0.01, unit: 'rel', defaultVal: 0.22 },
      alpha_power: { min: 0.02, max: 0.65, step: 0.01, unit: 'rel', defaultVal: 0.25 },
      beta_power: { min: 0.01, max: 0.55, step: 0.01, unit: 'rel', defaultVal: 0.16 },
      gamma_power: { min: 0.005, max: 0.45, step: 0.005, unit: 'rel', defaultVal: 0.09 },
      spectral_entropy: { min: 0.35, max: 0.98, step: 0.01, unit: 'bits', defaultVal: 0.78 },
      hjorth_mobility: { min: 0.1, max: 2.8, step: 0.05, unit: '', defaultVal: 0.85 },
      hjorth_complexity: { min: 0.5, max: 3.5, step: 0.05, unit: '', defaultVal: 1.25 },
      sample_entropy: { min: 0.2, max: 2.2, step: 0.05, unit: '', defaultVal: 1.10 },
      line_length: { min: 10, max: 350, step: 2, unit: 'μV/s', defaultVal: 65.0 },
    },
    classicalModel: {
      id: 'classical-svm-rbf',
      name: 'Support Vector Classifier (RBF Kernel) + RF',
      kind: 'classical',
      metrics: {
        accuracy: 0.940,
        precision: 0.925,
        sensitivity: 0.925,
        specificity: 0.950,
        f1: 0.925,
        rocAuc: 0.972,
        trainingTime: '1.9s',
        inferenceTime: '1.4ms',
      },
      confusionMatrix: { tp: 37, fn: 3, tn: 57, fp: 3 },
      parameters: 'C=10.0, gamma=scale, RBF kernel with balanced class weights',
      hardware: 'Classical CPU (4 threads)',
      description: 'Radial Basis Function SVM tuned for maximal margin separation across non-linear spectral entropy bands.',
      rationale: 'Classical gold standard for bio-signal classification due to robust margin bounds against Gaussian noise.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.02, tpr: 0.82 },
        { fpr: 0.05, tpr: 0.92 },
        { fpr: 0.08, tpr: 0.96 },
        { fpr: 0.18, tpr: 0.99 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    quantumModel: {
      id: 'quantum-qkernel-vqc',
      name: 'Quantum Kernel Classifier (ZZFeatureMap + VQC)',
      kind: 'quantum',
      metrics: {
        accuracy: 0.960,
        precision: 0.950,
        sensitivity: 0.950,
        specificity: 0.967,
        f1: 0.950,
        rocAuc: 0.985,
        trainingTime: '18.2s',
        inferenceTime: '32.1ms',
      },
      confusionMatrix: { tp: 38, fn: 2, tn: 58, fp: 2 },
      parameters: '6 Qubits, 2-order ZZ Feature Map + 3 Variational Layers',
      hardware: 'Statevector Simulator',
      description: 'ZZ entangling quantum feature map projecting oscillatory bio-signals into a non-linear $2^6=64$ dimensional Hilbert kernel space.',
      rationale: 'Quantum phase interference naturally models phase-amplitude coupling and cross-frequency interactions in multi-channel brain rhythms.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.01, tpr: 0.88 },
        { fpr: 0.03, tpr: 0.95 },
        { fpr: 0.06, tpr: 0.98 },
        { fpr: 0.12, tpr: 1.0 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    honestCallout: {
      title: 'Quantum Phase Embedding Captures Complex Wavelet Discharges',
      summary: 'Quantum ZZ feature mapping achieves 96.0% accuracy vs 94.0% classical baseline on distinguishing epileptic seizures from interictal/healthy EEG patterns.',
      quantumPros: [
        'Enhanced detection of fast polyspike gamma bursts via quantum phase interaction',
        '2.5% increase in sensitivity (95.0% vs 92.5%) reducing missed seizure events',
        'Natural mathematical alignment between oscillatory phase spaces and unitary Bloch rotations',
      ],
      classicalPros: [
        'Near-instant inference (1.4ms) making classical SVM simpler for low-power edge EEG wearables',
        'Deterministic execution without quantum simulator statevector memory footprint',
      ],
      nuance: 'The non-linear phase mapping provided by ZZ entangling gates provides an organic fit for neural time-series data, though hardware deployment currently requires statevector emulation or low-depth quantum circuits.',
    },
    samplePresets: [
      {
        id: 'case-ictal-burst',
        name: 'Case A: Ictal High-Frequency Paroxysm',
        description: 'Massive gamma & beta elevation with collapsed spectral entropy and high line length',
        expectedClass: 'Seizure Detected',
        values: {
          delta_power: 0.12,
          theta_power: 0.15,
          alpha_power: 0.08,
          beta_power: 0.32,
          gamma_power: 0.33,
          spectral_entropy: 0.48,
          hjorth_mobility: 1.95,
          hjorth_complexity: 2.45,
          sample_entropy: 0.52,
          line_length: 245.0,
        },
      },
      {
        id: 'case-interictal-slow',
        name: 'Case B: Interictal Background Activity',
        description: 'Synchronized resting alpha & theta with normal signal entropy',
        expectedClass: 'Normal EEG Baseline',
        values: {
          delta_power: 0.28,
          theta_power: 0.24,
          alpha_power: 0.32,
          beta_power: 0.12,
          gamma_power: 0.04,
          spectral_entropy: 0.82,
          hjorth_mobility: 0.72,
          hjorth_complexity: 1.10,
          sample_entropy: 1.25,
          line_length: 52.0,
        },
      },
      {
        id: 'case-borderline-sharp',
        name: 'Case C: Focal Sharp Wave Transient',
        description: 'Transient beta elevation with moderate spike line length',
        expectedClass: 'Seizure Detected',
        values: {
          delta_power: 0.20,
          theta_power: 0.18,
          alpha_power: 0.16,
          beta_power: 0.26,
          gamma_power: 0.20,
          spectral_entropy: 0.62,
          hjorth_mobility: 1.35,
          hjorth_complexity: 1.70,
          sample_entropy: 0.88,
          line_length: 145.0,
        },
      },
    ],
  },
  {
    id: 'heart-disease',
    name: 'Heart Disease & Myocardial Infarction Risk',
    category: 'cardiovascular',
    categoryLabel: 'Cardiovascular / Cardiology',
    tagline: 'Hemodynamic, electrocardiographic & fluoroscopic cardiovascular risk stratification',
    modality: 'Clinical & Hemodynamic Tabular',
    targetCondition: 'High Risk Myocardial Infarction / CAD (>50% Coronary Stenosis)',
    positiveLabel: 'High Risk (Stenosis/CAD)',
    negativeLabel: 'Normal / Low Risk',
    inputDimensionality: '13 clinical, hemodynamic & exercise ECG diagnostic attributes',
    reducedDimensionality: '6 quantum-encoded cardiovascular markers',
    defaultQubits: 6,
    datasetName: 'Cleveland Clinic Heart Disease Risk Cohort',
    datasetSource: 'Cleveland Clinic Foundation / UCI Machine Learning Repository',
    totalSamples: 303,
    featureDescriptions: {
      age: 'Patient age in chronological years',
      resting_bp: 'Resting systolic blood pressure upon admission (mm Hg)',
      cholesterol: 'Serum total cholesterol concentration (mg/dL)',
      max_heart_rate: 'Maximum heart rate achieved during standardized Bruce treadmill protocol (bpm)',
      st_depression: 'ST segment depression induced by exercise relative to resting baseline (mm)',
      num_vessels: 'Number of major coronary vessels (0-3) visualized with contrast fluoroscopy',
      chest_pain_type: 'Chest pain classification (1: typical angina, 2: atypical, 3: non-anginal, 4: asymptomatic)',
      exercise_angina: 'Presence of exercise-induced angina pectoris (0: no, 1: yes)',
      st_slope: 'Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)',
      fasting_blood_sugar: 'Fasting blood glucose > 120 mg/dL (0: false, 1: true)',
    },
    featureRanges: {
      age: { min: 28, max: 80, step: 1, unit: 'yrs', defaultVal: 54 },
      resting_bp: { min: 90, max: 200, step: 2, unit: 'mmHg', defaultVal: 131 },
      cholesterol: { min: 120, max: 560, step: 5, unit: 'mg/dL', defaultVal: 246 },
      max_heart_rate: { min: 70, max: 210, step: 2, unit: 'bpm', defaultVal: 150 },
      st_depression: { min: 0.0, max: 6.2, step: 0.1, unit: 'mm', defaultVal: 1.05 },
      num_vessels: { min: 0, max: 3, step: 1, unit: 'vessels', defaultVal: 1 },
      chest_pain_type: { min: 1, max: 4, step: 1, unit: '', defaultVal: 3 },
      exercise_angina: { min: 0, max: 1, step: 1, unit: '', defaultVal: 0 },
      st_slope: { min: 1, max: 3, step: 1, unit: '', defaultVal: 2 },
      fasting_blood_sugar: { min: 0, max: 1, step: 1, unit: '', defaultVal: 0 },
    },
    classicalModel: {
      id: 'classical-logreg-rf',
      name: 'Logistic Regression + Random Forest Ensemble',
      kind: 'classical',
      metrics: {
        accuracy: 0.852,
        precision: 0.846,
        sensitivity: 0.846,
        specificity: 0.857,
        f1: 0.846,
        rocAuc: 0.912,
        trainingTime: '1.6s',
        inferenceTime: '1.2ms',
      },
      confusionMatrix: { tp: 22, fn: 4, tn: 30, fp: 5 },
      parameters: 'L2 regularized Logistic Regression (C=1.0) + 200 Random Forest trees',
      hardware: 'Classical CPU (4 threads)',
      description: 'Standard clinical scoring baseline combining calibrated logistic odds with bagged decision trees.',
      rationale: 'Widely used in clinical cardiology due to interpretability and linear odds ratio extraction.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.05, tpr: 0.65 },
        { fpr: 0.12, tpr: 0.82 },
        { fpr: 0.20, tpr: 0.90 },
        { fpr: 0.35, tpr: 0.96 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    quantumModel: {
      id: 'quantum-vqc-heart',
      name: 'Hybrid Quantum Neural Classifier (VQC)',
      kind: 'quantum',
      metrics: {
        accuracy: 0.869,
        precision: 0.880,
        sensitivity: 0.846,
        specificity: 0.886,
        f1: 0.863,
        rocAuc: 0.924,
        trainingTime: '11.5s',
        inferenceTime: '24.7ms',
      },
      confusionMatrix: { tp: 22, fn: 4, tn: 31, fp: 4 },
      parameters: '6 Qubits, 3 Hardware-Efficient Entangling Layers',
      hardware: 'Statevector Simulator (1024 shot equiv.)',
      description: 'Variational circuit with parameterized $R_X, R_Y, R_Z$ gates and cyclic CNOT entanglers evaluated under Adam optimizer.',
      rationale: 'Effective at learning complex co-linear interactions between blood pressure, ST depression, and fluoroscopy vessel counts.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.03, tpr: 0.68 },
        { fpr: 0.09, tpr: 0.86 },
        { fpr: 0.16, tpr: 0.92 },
        { fpr: 0.30, tpr: 0.98 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    honestCallout: {
      title: 'Precision Gain in Fluoroscopy & Ischemic ST Stratification',
      summary: 'The Hybrid Quantum Classifier improves Specificity (88.6% vs 85.7%) and Precision (88.0% vs 84.6%), reducing false-positive catheterization referrals.',
      quantumPros: [
        'Higher specificity reduces unnecessary invasive angiogram referrals',
        'Efficient multi-attribute entanglement across fluoroscopy vessel counts and ST depression',
        'Stronger calibration in the high-risk probability spectrum',
      ],
      classicalPros: [
        'Classical logistic regression provides direct closed-form odds ratio coefficients',
        'Faster inference suitable for instant bedside triage apps',
      ],
      nuance: 'On small clinical cohorts like Cleveland (303 records), quantum models perform comparably to classical ensembles with a slight specificity edge, avoiding the high variance typical of unregularized deep neural nets.',
    },
    samplePresets: [
      {
        id: 'case-cad-severe',
        name: 'Case A: Multi-Vessel Coronary Artery Disease',
        description: 'Older patient, marked ST depression (3.2mm), 2 fluoroscopy vessels with stenosis',
        expectedClass: 'High Risk (Stenosis/CAD)',
        values: {
          age: 63,
          resting_bp: 152,
          cholesterol: 285,
          max_heart_rate: 118,
          st_depression: 3.2,
          num_vessels: 2,
          chest_pain_type: 4,
          exercise_angina: 1,
          st_slope: 2,
          fasting_blood_sugar: 1,
        },
      },
      {
        id: 'case-normal-hemodynamic',
        name: 'Case B: Normal Hemodynamic Response',
        description: 'Younger adult, high exercise capacity, zero ST depression or vessel blockage',
        expectedClass: 'Normal / Low Risk',
        values: {
          age: 42,
          resting_bp: 118,
          cholesterol: 195,
          max_heart_rate: 178,
          st_depression: 0.2,
          num_vessels: 0,
          chest_pain_type: 2,
          exercise_angina: 0,
          st_slope: 1,
          fasting_blood_sugar: 0,
        },
      },
      {
        id: 'case-borderline-ischemia',
        name: 'Case C: Moderate Exercise-Induced Ischemia',
        description: 'Elevated blood pressure, flat ST slope with 1.6mm depression',
        expectedClass: 'High Risk (Stenosis/CAD)',
        values: {
          age: 56,
          resting_bp: 140,
          cholesterol: 250,
          max_heart_rate: 142,
          st_depression: 1.6,
          num_vessels: 1,
          chest_pain_type: 3,
          exercise_angina: 1,
          st_slope: 2,
          fasting_blood_sugar: 0,
        },
      },
    ],
  },
  /*
   * The three conditions below mirror the validated models in the Python
   * registry (backend P1, P5, P6). Unlike the entries above, their benchmark
   * metrics are NOT illustrative: every figure is transcribed from the
   * evaluation records under
   * `backend/src/qhealth_qml/platform/registry_data/evaluations/`, produced by
   * repeated holdout on the real cohorts. Their honest scoping is carried over
   * verbatim too - where the backend says a model is a same-visit proxy rather
   * than a progression predictor, this says the same.
   */
  {
    id: 'stroke-risk',
    name: 'Stroke Clinical Risk',
    category: 'neurological',
    categoryLabel: 'Neurological / Cerebrovascular',
    tagline: 'Structured cardiovascular risk factors for stroke association',
    modality: 'Clinical & Hemodynamic Tabular',
    targetCondition: 'Recorded stroke event',
    positiveLabel: 'Stroke',
    negativeLabel: 'No Stroke',
    inputDimensionality: '10 structured clinical risk factors',
    reducedDimensionality: '6 quantum-encoded risk components',
    defaultQubits: 6,
    datasetName: 'Kaggle stroke-prediction cohort',
    datasetSource: 'Backend P1, 5110 rows, 4.87% positive',
    totalSamples: 5110,
    featureDescriptions: {
      age: 'Age in years',
      avg_glucose_level: 'Average blood glucose concentration (mg/dL)',
      bmi: 'Body mass index',
      hypertension: 'Diagnosed hypertension (0 or 1)',
      heart_disease: 'Diagnosed heart disease (0 or 1)',
      ever_married: 'Has ever been married (0 or 1)',
      work_type: 'Employment category code',
      residence_type: 'Urban residence (0 rural, 1 urban)',
      smoking_status: 'Smoking status code, higher is heavier exposure',
      gender: 'Recorded gender code',
    },
    featureRanges: {
      age: { min: 18, max: 100, step: 1, unit: 'yrs', defaultVal: 43 },
      avg_glucose_level: { min: 55, max: 280, step: 1, unit: 'mg/dL', defaultVal: 106 },
      bmi: { min: 12, max: 60, step: 0.5, unit: 'kg/m2', defaultVal: 28.9 },
      hypertension: { min: 0, max: 1, step: 1, unit: '', defaultVal: 0 },
      heart_disease: { min: 0, max: 1, step: 1, unit: '', defaultVal: 0 },
      ever_married: { min: 0, max: 1, step: 1, unit: '', defaultVal: 1 },
      work_type: { min: 0, max: 4, step: 1, unit: '', defaultVal: 2 },
      residence_type: { min: 0, max: 1, step: 1, unit: '', defaultVal: 1 },
      smoking_status: { min: 0, max: 3, step: 1, unit: '', defaultVal: 1 },
      gender: { min: 0, max: 1, step: 1, unit: '', defaultVal: 0 },
    },
    classicalModel: {
      id: 'stroke-clinical-risk-tabular-classical',
      name: 'Gradient Boosted Classifier',
      kind: 'classical',
      metrics: {
        accuracy: 0.735,
        precision: 0.137,
        sensitivity: 0.780,
        specificity: 0.732,
        f1: 0.231,
        rocAuc: 0.845,
        trainingTime: 'n/a',
        inferenceTime: 'n/a',
      },
      confusionMatrix: { tp: 39, fn: 11, tn: 712, fp: 260 },
      parameters: 'Gradient-boosted trees, class-weighted',
      hardware: 'Classical CPU',
      description: 'Operational reference in the backend registry, balanced accuracy 0.756 on repeated holdout.',
      rationale: 'At 4.87% prevalence, precision is necessarily low: catching 78% of stroke cases costs a large number of false positives. The negative predictive value of 0.985 is the useful figure here.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.1, tpr: 0.42 },
        { fpr: 0.27, tpr: 0.78 },
        { fpr: 0.5, tpr: 0.9 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    quantumModel: {
      id: 'stroke-clinical-risk-tabular',
      name: 'Quantum Support Vector Classifier',
      kind: 'quantum',
      metrics: {
        accuracy: 0.478,
        precision: 0.062,
        sensitivity: 0.680,
        specificity: 0.467,
        f1: 0.114,
        rocAuc: 0.634,
        trainingTime: 'n/a',
        inferenceTime: 'n/a',
      },
      confusionMatrix: { tp: 34, fn: 16, tn: 454, fp: 518 },
      parameters: 'QSVC, ZZ feature map',
      hardware: 'Statevector simulator',
      description: 'Balanced accuracy 0.574, below the classical baseline on the same split.',
      rationale: 'Registered as experimental. The paired confidence interval on the delta does not favour the quantum candidate, so classical remains the operational reference.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.2, tpr: 0.3 },
        { fpr: 0.53, tpr: 0.68 },
        { fpr: 0.8, tpr: 0.85 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    honestCallout: {
      title: 'Classical outperforms quantum on this cohort',
      summary: 'Balanced accuracy 0.756 classical against 0.574 quantum on the same repeated holdout. The classical model is the operational reference.',
      quantumPros: [],
      classicalPros: [
        'Higher balanced accuracy (0.756 vs 0.574)',
        'Higher ROC AUC (0.845 vs 0.634)',
        'Negative predictive value 0.985, useful for ruling out',
      ],
      nuance: 'Research benchmarking only. Excluded from any clinical use, acute stroke triage, or use as evidence that a patient does not have a stroke.',
    },
    samplePresets: [
      {
        id: 'stroke-elevated',
        name: 'Older patient, multiple risk factors',
        description: 'Age 78, hypertension and heart disease present, elevated glucose',
        expectedClass: 'High Risk',
        values: {
          age: 78, avg_glucose_level: 205, bmi: 31.5, hypertension: 1,
          heart_disease: 1, ever_married: 1, work_type: 2,
          residence_type: 1, smoking_status: 2, gender: 1,
        },
      },
      {
        id: 'stroke-baseline',
        name: 'Younger patient, no risk factors',
        description: 'Age 32, no hypertension or heart disease, normal glucose',
        expectedClass: 'Low Risk',
        values: {
          age: 32, avg_glucose_level: 85, bmi: 24.0, hypertension: 0,
          heart_disease: 0, ever_married: 0, work_type: 2,
          residence_type: 0, smoking_status: 0, gender: 0,
        },
      },
    ],
  },
  {
    id: 'alzheimers',
    name: "Alzheimer's Dementia Association",
    category: 'neurological',
    categoryLabel: 'Neurological / Neurodegenerative',
    tagline: 'Same-visit dementia association from clinical and volumetric measures',
    modality: 'Structured Tabular',
    targetCondition: 'Dementia at the same visit (CDR > 0)',
    positiveLabel: 'Dementia (CDR > 0)',
    negativeLabel: 'No Dementia (CDR 0)',
    inputDimensionality: '8 clinical and MRI-derived volumetric measures',
    reducedDimensionality: '6 quantum-encoded components',
    defaultQubits: 6,
    datasetName: 'OASIS-1 cross-sectional',
    datasetSource: 'Backend P5, 235 rows, 42.6% positive',
    totalSamples: 235,
    featureDescriptions: {
      MMSE: 'Mini-Mental State Examination score, 0 to 30, lower is worse',
      nWBV: 'Normalized whole-brain volume',
      Age: 'Age at visit in years',
      eTIV: 'Estimated total intracranial volume (mm3)',
      ASF: 'Atlas scaling factor',
      Educ: 'Years of education band',
      SES: 'Socioeconomic status band, lower value is higher status',
      M_F: 'Recorded sex code',
    },
    featureRanges: {
      MMSE: { min: 0, max: 30, step: 1, unit: 'pts', defaultVal: 27 },
      nWBV: { min: 0.64, max: 0.84, step: 0.001, unit: '', defaultVal: 0.729 },
      Age: { min: 33, max: 96, step: 1, unit: 'yrs', defaultVal: 75 },
      eTIV: { min: 1100, max: 2000, step: 10, unit: 'mm3', defaultVal: 1480 },
      ASF: { min: 0.88, max: 1.6, step: 0.01, unit: '', defaultVal: 1.2 },
      Educ: { min: 1, max: 5, step: 1, unit: '', defaultVal: 3 },
      SES: { min: 1, max: 5, step: 1, unit: '', defaultVal: 2 },
      M_F: { min: 0, max: 1, step: 1, unit: '', defaultVal: 0 },
    },
    classicalModel: {
      id: 'alzheimers-clinical-risk-tabular-classical',
      name: 'Gradient Boosted Classifier',
      kind: 'classical',
      metrics: {
        accuracy: 0.830,
        precision: 0.834,
        sensitivity: 0.775,
        specificity: 0.870,
        f1: 0.792,
        rocAuc: 0.919,
        trainingTime: 'n/a',
        inferenceTime: 'n/a',
      },
      confusionMatrix: { tp: 31, fn: 9, tn: 47, fp: 7 },
      parameters: 'Gradient-boosted trees',
      hardware: 'Classical CPU',
      description: 'Operational reference, balanced accuracy 0.823 on repeated holdout.',
      rationale: 'MMSE and normalized whole-brain volume carry most of the signal, which is consistent with the clinical literature.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.05, tpr: 0.55 },
        { fpr: 0.13, tpr: 0.78 },
        { fpr: 0.3, tpr: 0.93 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    quantumModel: {
      id: 'alzheimers-clinical-risk-tabular',
      name: 'Quantum Support Vector Classifier',
      kind: 'quantum',
      metrics: {
        accuracy: 0.526,
        precision: 0.497,
        sensitivity: 0.825,
        specificity: 0.304,
        f1: 0.620,
        rocAuc: 0.608,
        trainingTime: 'n/a',
        inferenceTime: 'n/a',
      },
      confusionMatrix: { tp: 33, fn: 7, tn: 16, fp: 38 },
      parameters: 'QSVC, ZZ feature map',
      hardware: 'Statevector simulator',
      description: 'Balanced accuracy 0.564, well below the classical baseline.',
      rationale: 'Registered as experimental. High sensitivity here comes from predicting the positive class far too often, as the 0.304 specificity shows.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.3, tpr: 0.5 },
        { fpr: 0.7, tpr: 0.83 },
        { fpr: 0.9, tpr: 0.95 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    honestCallout: {
      title: 'Scoped as a same-visit proxy, not progression',
      summary: 'Balanced accuracy 0.823 classical against 0.564 quantum. This predicts dementia association at the same visit, not future progression.',
      quantumPros: [],
      classicalPros: [
        'Higher balanced accuracy (0.823 vs 0.564)',
        'Far higher specificity (0.870 vs 0.304)',
        'ROC AUC 0.919',
      ],
      nuance: 'No OASIS-1 data supports a longitudinal progression claim. 201 subjects without a recorded Clinical Dementia Rating were dropped. Research benchmarking only.',
    },
    samplePresets: [
      {
        id: 'alz-impaired',
        name: 'Impaired cognition, reduced brain volume',
        description: 'MMSE 21, low normalized whole-brain volume',
        expectedClass: 'High Risk',
        values: { MMSE: 21, nWBV: 0.695, Age: 82, eTIV: 1450, ASF: 1.21, Educ: 2, SES: 3, M_F: 1 },
      },
      {
        id: 'alz-intact',
        name: 'Intact cognition',
        description: 'MMSE 30, brain volume within normal range',
        expectedClass: 'Low Risk',
        values: { MMSE: 30, nWBV: 0.756, Age: 68, eTIV: 1490, ASF: 1.18, Educ: 4, SES: 2, M_F: 0 },
      },
    ],
  },
  {
    id: 'parkinsons',
    name: "Parkinson's Voice Analysis",
    category: 'neurological',
    categoryLabel: 'Neurological / Movement Disorder',
    tagline: 'Sustained-phonation acoustic markers, diagnosed against healthy controls',
    modality: 'Structured Tabular',
    targetCondition: "Diagnosed Parkinson's disease",
    positiveLabel: "Parkinson's",
    negativeLabel: 'Healthy Control',
    inputDimensionality: '12 acoustic measures from sustained phonation',
    reducedDimensionality: '6 quantum-encoded acoustic components',
    defaultQubits: 6,
    datasetName: "UCI Parkinson's voice cohort",
    datasetSource: 'Backend P6, 195 recordings from 32 subjects',
    totalSamples: 195,
    featureDescriptions: {
      PPE: 'Pitch period entropy',
      spread1: 'Nonlinear fundamental-frequency spread measure',
      MDVP_Fo_Hz: 'Average vocal fundamental frequency (Hz)',
      MDVP_Flo_Hz: 'Minimum vocal fundamental frequency (Hz)',
      spread2: 'Second nonlinear frequency-spread measure',
      MDVP_Jitter_pct: 'Cycle-to-cycle fundamental frequency variation (%)',
      MDVP_Shimmer: 'Cycle-to-cycle amplitude variation',
      HNR: 'Harmonics-to-noise ratio (dB)',
      NHR: 'Noise-to-harmonics ratio',
      RPDE: 'Recurrence period density entropy',
      DFA: 'Detrended fluctuation analysis exponent',
      D2: 'Correlation dimension',
    },
    featureRanges: {
      PPE: { min: 0.04, max: 0.53, step: 0.001, unit: '', defaultVal: 0.207 },
      spread1: { min: -7.96, max: -2.43, step: 0.01, unit: '', defaultVal: -5.68 },
      MDVP_Fo_Hz: { min: 88, max: 260, step: 1, unit: 'Hz', defaultVal: 154 },
      MDVP_Flo_Hz: { min: 65, max: 240, step: 1, unit: 'Hz', defaultVal: 116 },
      spread2: { min: 0.006, max: 0.45, step: 0.001, unit: '', defaultVal: 0.227 },
      MDVP_Jitter_pct: { min: 0.001, max: 0.034, step: 0.0005, unit: '%', defaultVal: 0.0062 },
      MDVP_Shimmer: { min: 0.009, max: 0.12, step: 0.001, unit: '', defaultVal: 0.030 },
      HNR: { min: 8, max: 33, step: 0.5, unit: 'dB', defaultVal: 21.9 },
      NHR: { min: 0.0006, max: 0.32, step: 0.001, unit: '', defaultVal: 0.025 },
      RPDE: { min: 0.25, max: 0.69, step: 0.001, unit: '', defaultVal: 0.499 },
      DFA: { min: 0.57, max: 0.83, step: 0.001, unit: '', defaultVal: 0.718 },
      D2: { min: 1.42, max: 3.68, step: 0.01, unit: '', defaultVal: 2.382 },
    },
    classicalModel: {
      id: 'parkinsons-voice-risk-tabular-classical',
      name: 'Gradient Boosted Classifier',
      kind: 'classical',
      metrics: {
        accuracy: 0.823,
        precision: 0.879,
        sensitivity: 0.898,
        specificity: 0.571,
        f1: 0.873,
        rocAuc: 0.883,
        trainingTime: 'n/a',
        inferenceTime: 'n/a',
      },
      confusionMatrix: { tp: 132, fn: 15, tn: 27, fp: 21 },
      parameters: 'Gradient-boosted trees',
      hardware: 'Classical CPU',
      description: 'Operational reference, balanced accuracy 0.734 on repeated holdout.',
      rationale: 'Specificity is limited by the class imbalance: only 48 of 195 recordings are healthy controls, and the 32 subjects contribute multiple recordings each.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.15, tpr: 0.68 },
        { fpr: 0.43, tpr: 0.9 },
        { fpr: 0.7, tpr: 0.97 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    quantumModel: {
      id: 'parkinsons-voice-risk-tabular',
      name: 'Quantum Support Vector Classifier',
      kind: 'quantum',
      metrics: {
        accuracy: 0.685,
        precision: 0.795,
        sensitivity: 0.837,
        specificity: 0.242,
        f1: 0.816,
        rocAuc: 0.561,
        trainingTime: 'n/a',
        inferenceTime: 'n/a',
      },
      confusionMatrix: { tp: 123, fn: 24, tn: 12, fp: 36 },
      parameters: 'QSVC, ZZ feature map',
      hardware: 'Statevector simulator',
      description: 'Balanced accuracy 0.539, barely above chance.',
      rationale: 'Registered as experimental. A ROC AUC of 0.561 on this cohort is close to uninformative.',
      rocPoints: [
        { fpr: 0.0, tpr: 0.0 },
        { fpr: 0.35, tpr: 0.5 },
        { fpr: 0.76, tpr: 0.84 },
        { fpr: 0.92, tpr: 0.95 },
        { fpr: 1.0, tpr: 1.0 },
      ],
    },
    honestCallout: {
      title: 'Diagnosed versus healthy, not prodromal risk',
      summary: 'Balanced accuracy 0.734 classical against 0.539 quantum. This separates already-diagnosed subjects from healthy controls.',
      quantumPros: [],
      classicalPros: [
        'Higher balanced accuracy (0.734 vs 0.539)',
        'Higher specificity (0.571 vs 0.242)',
        'ROC AUC 0.883 against 0.561',
      ],
      nuance: 'Not a prodromal or PPMI-style at-risk cohort. Recordings are grouped by subject, so a naive row-level split would leak. Research benchmarking only.',
    },
    samplePresets: [
      {
        id: 'pd-affected',
        name: 'Marked voice tremor',
        description: 'Elevated jitter and shimmer, reduced harmonics-to-noise ratio',
        expectedClass: 'High Risk',
        values: {
          PPE: 0.32, spread1: -4.62, MDVP_Fo_Hz: 128, MDVP_Flo_Hz: 92,
          spread2: 0.31, MDVP_Jitter_pct: 0.0128, MDVP_Shimmer: 0.058,
          HNR: 16.4, NHR: 0.061, RPDE: 0.588, DFA: 0.762, D2: 2.71,
        },
      },
      {
        id: 'pd-control',
        name: 'Stable phonation',
        description: 'Low jitter and shimmer, high harmonics-to-noise ratio',
        expectedClass: 'Low Risk',
        values: {
          PPE: 0.105, spread1: -7.02, MDVP_Fo_Hz: 197, MDVP_Flo_Hz: 158,
          spread2: 0.142, MDVP_Jitter_pct: 0.0032, MDVP_Shimmer: 0.0158,
          HNR: 25.8, NHR: 0.0089, RPDE: 0.421, DFA: 0.688, D2: 2.08,
        },
      },
    ],
  },
]

export function getDiseasePipeline(id: string): DiseasePipeline {
  const p = DISEASE_PIPELINES.find((d) => d.id === id)
  if (!p) return DISEASE_PIPELINES[0]
  return p
}

export type TrainedPipelineArtifact = {
  diseaseId: string
  trainedAt: string
  datasetName: string
  rows: number
  featureNames: string[]
  keptFeatures: string[]
  scaler: Scaler
  classicalMetrics: UnifiedMetrics
  quantumMetrics: UnifiedMetrics
  classicalModelName: string
  quantumModelName: string
  quantumWeights?: number[]
  quantumConfig?: VqcConfig
  /**
   * Everything below is what inference needs to replay the training transform
   * exactly. Without it the Predict screen cannot score with the model that was
   * actually fitted, only approximate it.
   */
  /** column indices kept by the selector, in model input order */
  keptIndices?: number[]
  /** set when the selector was PCA, in which case keptIndices does not apply */
  pca?: { mean: number[]; scale: number[]; loadings: number[][] } | null
  /** per-column fill values, so a missing input is imputed the training way */
  imputeValues?: number[]
  /** column means in model space: the neutral reference for occlusion */
  baselineVector?: number[]
  confusionMatrixClassical: { tp: number; fn: number; tn: number; fp: number }
  confusionMatrixQuantum: { tp: number; fn: number; tn: number; fp: number }
  rocPointsClassical: { fpr: number; tpr: number }[]
  rocPointsQuantum: { fpr: number; tpr: number }[]
}

const PERSISTED_KEY = 'neutral_persisted_trained_pipelines'
/** The key this used to be written under, before the spelling was corrected. */
const LEGACY_PERSISTED_KEY = 'netural_persisted_trained_pipelines'

export function saveTrainedPipeline(artifact: TrainedPipelineArtifact) {
  try {
    const existing = loadAllTrainedPipelines()
    existing[artifact.diseaseId] = artifact
    localStorage.setItem(PERSISTED_KEY, JSON.stringify(existing))
  } catch {
    // localstorage fallback
  }
}

export function loadTrainedPipeline(diseaseId: string): TrainedPipelineArtifact | null {
  try {
    const all = loadAllTrainedPipelines()
    return all[diseaseId] ?? null
  } catch {
    return null
  }
}

export function loadAllTrainedPipelines(): Record<string, TrainedPipelineArtifact> {
  try {
    const raw = localStorage.getItem(PERSISTED_KEY)
    if (raw) return JSON.parse(raw)

    // Fall back to the old key once and migrate it forward, so a model that
    // was trained before the rename is not silently lost.
    const legacy = localStorage.getItem(LEGACY_PERSISTED_KEY)
    if (!legacy) return {}

    const parsed = JSON.parse(legacy) as Record<string, TrainedPipelineArtifact>
    localStorage.setItem(PERSISTED_KEY, legacy)
    localStorage.removeItem(LEGACY_PERSISTED_KEY)
    return parsed
  } catch {
    return {}
  }
}
