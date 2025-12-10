# SchrödingerSwap: Project Roadmap & Tasks

**Objective:** Build a Quantum Risk Engine for Islamic Cross-Currency Swaps.
**Architecture:** Hybrid Quantum-Classical (Rust Core + Python Orchestrator).
**Current Phase:** Phase 1 (Foundations).

---

## 📅 Month 1: Classical Foundations & Data Infrastructure
*Goal: Establish the "Ground Truth" using classical financial engineering before introducing quantum elements.*

### 📚 Reading Dependencies (Mathematical Basis)
- [ ] **Mastering CVA & WWR:** Read *Counterparty Credit Risk* by Brigo et al. (Chapters 4-6).
    - *Focus:* Derivation of CVA and Wrong-Way Risk (WWR) correlation.
- [ ] **The Benchmark Model:** Read *Pykhtin (2005) "A Guide to Modelling Counterparty Credit Risk"*.
    - *Focus:* Semi-analytical approximations for WWR to serve as the unit test baseline.
- [ ] **Islamic Finance Constraints:** Read *Jobst (2007) "The Economics of Islamic Finance"*.
    - *Focus:* The "Regulatory Floor" logic for Hamish Jiddiyyah.

### 🛠️ Engineering Tasks (Implementation)
- [ ] **Infrastructure Setup**
    - [ ] Initialize React + TypeScript Frontend (Dealing Desk UI).
    - [ ] Initialize Python FastAPI Backend (Orchestrator).
    - [ ] Deploy TimescaleDB for financial time-series persistence.
- [ ] **Market Data Ingestor**
    - [ ] Implement FX Spot Rate fetcher.
    - [ ] Implement CDS Spread fetcher (Credit Health Proxy).
- [ ] **Classical Pre-processor**
    - [ ] Implement `CorrelationMatrixBuilder` to derive $\lambda$ from FX/CDS history.
    - [ ] **Test:** Verify output against Pykhtin's analytical approximation.

---

## 📅 Month 2: The Quantum Core (Hamiltonian Simulation)
*Goal: Build the Rust engine capable of simulating the time-evolution of the financial state.*

### 📚 Reading Dependencies (Mathematical Basis)
- [ ] **Unitary Operators:** Read *Nielsen & Chuang* (Chapters 4 & 8).
    - *Focus:* How to map the Hamiltonian to Unitary gates.
- [ ] **Time Evolution:** Read *Suzuki (1991)* on Trotter-Suzuki decomposition.
    - *Focus:* Breaking $e^{-iHt}$ into discrete gates.
- [ ] **Tensor Networks:** Read *Schollwöck (2011)* and *Orús (2014)*.
    - *Focus:* Matrix Product States (MPS) and SVD truncation for GPU simulation.

### 🛠️ Engineering Tasks (Implementation)
- [ ] **Rust Quantum Engine**
    - [ ] Define structs for Registers: `|FX>` and `|Credit>`.
    - [ ] Implement `HamiltonianBuilder` trait.
    - [ ] Code the Interaction Term ($H_{int}$) logic using the correlation $\lambda$.
- [ ] **Simulation Logic (MPS)**
    - [ ] Implement Matrix Product State (MPS) representation.
    - [ ] Implement SVD truncation (bond dimension optimization).
    - [ ] Implement Trotter Step time evolution ($e^{-iHt}$).

---

## 📅 Month 3: Quantum Risk Algorithms (QAE)
*Goal: Implement the "Holy Grail" algorithm to estimate Tail Risk (CVaR).*

### 📚 Reading Dependencies (Mathematical Basis)
- [ ] **The Core Algorithm:** Read *Woerner & Egger (2019) "Quantum risk analysis"*.
    - *Focus:* Mapping prob distributions to amplitudes and QAE execution.
- [ ] **State Preparation:** Read *Grover & Rudolph (2002)*.
    - *Focus:* Efficient loading of log-concave distributions.
- [ ] **Amplitude Estimation:** Read *Brassard et al. (2002)*.
    - *Focus:* The quadratic speedup mechanism.

### 🛠️ Engineering Tasks (Implementation)
- [ ] **Quantum State Loading**
    - [ ] Implement `VolatilitySurfaceLoader` (Grover-Rudolph method).
- [ ] **Amplitude Estimation (QAE)**
    - [ ] Implement the Oracle $O$ to flag "Disaster States" (High FX Exposure + Default).
    - [ ] Implement the Grover Operator $Q$.
    - [ ] **Simulation:** Run QAE on NVIDIA cuQuantum backend.
- [ ] **Benchmarking**
    - [ ] Compare convergence rate $O(M^{-1})$ vs Classical Monte Carlo $O(M^{-1/2})$.

---

## 📅 Month 4: Compliance, Integration & MVP
*Goal: Finalize the "Hamish Jiddiyyah" calculator and ensure Sharia compliance.*

### 📚 Reading Dependencies (Context)
- [ ] **Hardware Noise:** Read *Sheldon et al. (2016)*.
    - *Focus:* Cross-talk errors for robustness checking.
- [ ] **Sharia Logic:** Re-read *Jobst (2007)* regarding risk-sharing vs risk-transfer.

### 🛠️ Engineering Tasks (Implementation)
- [ ] **Sharia Guardrails**
    - [ ] Implement Logic Gate: Reject any input based on LIBOR/SOFR.
    - [ ] Implement `HamishJiddiyyah` calculator: $Max(Model_{99\%}, Floor)$.
- [ ] **Auditability**
    - [ ] Integrate Hyperledger Fabric to hash Hamiltonian parameters.
- [ ] **Frontend Visualization**
    - [ ] Implement D3.js Risk Surface Plot (FX Rate vs Default Prob).
