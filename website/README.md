# QTensor Engine · Production Web Portal (`qtensor.com.au`)

The official production web portal and institutional diligence showcase for the **QTensor Engine** — an analytical, training-free LLM compression engine compiling frontier foundation models into sub-3GB edge silicon binaries.

* **Live Domain:** [qtensor.com.au](https://qtensor.com.au)
* **Parent Firm:** [ZeroFold Engineering Practice](https://zerofold.com.au)
* **Founder & Lead Architect:** [Trent Parsons](https://www.linkedin.com/in/trent-i-parsons/)
* **Gated Evaluation Weights:** [🤗 trentzap/QTensor-7B-Enterprise-Edge](https://huggingface.co/trentzap/QTensor-7B-Enterprise-Edge)

---

## ⚡ Architecture & Technology Stack

The web portal is built with **Astro 4.x** and **Tailwind CSS**, pre-compiled to static HTML/CSS/JS for sub-millisecond initial load times, zero-runtime overhead, and deterministic edge CDN delivery.

* **Framework:** [Astro](https://astro.build/) (`output: 'static'`)
* **Styling:** Custom ZeroFold Minimalist Design System (`#f5f2ea` paper palette, `#0b0b0b` high-contrast ink, JetBrains Mono & Inter typography)
* **Deploy Target:** GitHub Pages with automated CI/CD via GitHub Actions (`.github/workflows/deploy.yml`)
* **Analytics & Telemetry:** Air-gapped static assets, zero tracker bloat, strict privacy compliance.

---

## 🧭 Page Routes & Site Structure

```text
website/
├── public/
│   ├── favicon.ico                   # Production brand icon
│   ├── favicon.svg                   # Vector SVG brand glyph
│   └── robots.txt                    # Search indexing rules
├── src/
│   ├── env.d.ts                      # TypeScript definitions
│   └── pages/
│       ├── index.astro               # Main product showcase, silicon telemetry & simulator
│       ├── contact.astro             # Leadership contact, diligence channels & practice links
│       └── contacts.astro            # Canonical redirect to /contact
├── astro.config.mjs                  # Site definition (https://qtensor.com.au)
├── package.json                      # Build scripts & dependencies
├── tailwind.config.mjs               # ZeroFold design tokens & palette
└── tsconfig.json                     # Strict TypeScript configuration
```

---

## 📊 Core Features & Interactive Elements

1. **Flagship 7B Silicon Telemetry:**
   * **Memory Footprint:** 3.02 GB VRAM (sub-4GB edge barrier broken).
   * **Inference Latency:** < 8ms Time to First Token (TTFT).
   * **Intelligence Retention:** 54.88% HumanEval Pass@1 (50.61% HumanEval+).
2. **Interactive Compilation Simulator:**
   * In-browser live simulator demonstrating deterministic closed-form token generation.
3. **Three-Tier Topological Architecture:**
   * High-level architectural narrative detailing Dynamic Rank Subspace Decomposition, Subspace Bridge Compensation, and Scale-Invariant Buffer Zones.
4. **Institutional Funnels & Evaluation Access:**
   * Direct CTAs to the gated Hugging Face repository (`trentzap/QTensor-7B-Enterprise-Edge`).
   * "Request Technical Brief" modal for institutional diligence packs.
   * Direct founder channels linking to Trent Parsons' LinkedIn profile.
   * Direct links to ZeroFold's engineering practice (`zerofold.com.au` & `zerofold.co/contact`).

---

## 🛠️ Local Development & Build

All commands are run from the `website/` directory:

```bash
# 1. Install dependencies
npm install

# 2. Start local development server (runs on http://localhost:4321)
npm run dev

# 3. Compile static production bundle to ./dist/
npm run build

# 4. Preview compiled production build locally
npm run preview
```

---

## 🔒 Intellectual Property & Sanitization Policy

The website repository is strictly sanitized in compliance with ZeroFold's intellectual property and patent protection guidelines:
* **No Proprietary Code:** Absolutely zero `.py` compiler scripts, Triton `@triton.jit` kernels, or topological mapping files (`asvd_config.yaml`) are contained within this web repository.
* **Abstracted Metrics:** Public documentation utilizes high-level architectural terminology without disclosing raw internal buffer layer indices or dynamic rank multipliers.
* **Gated Evaluation:** Model evaluation weights hosted on Hugging Face are gated and distributed under executed Mutual Non-Disclosure Agreements (MNDA).

---

## 📞 Commercial Licensing & Institutional Contact

* **Product Portal:** [qtensor.com.au](https://qtensor.com.au)
* **ZeroFold Engineering Practice:** [zerofold.com.au](https://zerofold.com.au)
* **Direct Practice Contact:** [zerofold.co/contact](https://zerofold.co/contact)
* **Founder & Lead Architect:** [Trent Parsons (LinkedIn)](https://www.linkedin.com/in/trent-i-parsons/)
* **Gated Evaluation Funnel:** [Hugging Face Repository](https://huggingface.co/trentzap/QTensor-7B-Enterprise-Edge)

---

© 2026 QTensor Pty Ltd · Developed by ZeroFold · Patent Pending · All Rights Reserved.
