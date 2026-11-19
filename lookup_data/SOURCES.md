# Data Sources for `lookup` Reference Table

This document records the provenance, licensing, version, and retrieval metadata for all reference data compiled into the Reckoner Harness (`reckoner-harness`) `lookup` tool.

All reference datasets included in this tool are unencumbered open public facts, works of the United States Government (17 U.S.C. § 105), NIST open reference data, or international public scientific standards. No CC BY-SA, restrictive, or proprietary data is included.

---

## 1. Physical & Chemical Constants: NIST CODATA 2022

- **Source**: National Institute of Standards and Technology (NIST), Physical Measurement Laboratory.
- **Title**: Fundamental Physical Constants: Complete Listing (2022 CODATA adjustment).
- **URL**: `https://physics.nist.gov/cuu/Constants/Table/allascii.txt`
- **Version**: 2022 CODATA adjustment (published May 2024).
- **Retrieval Date**: 2026-10-01.
- **License / Terms of Use**: NIST Standard Reference Database 121 (CODATA Fundamental Physical Constants). Under the Standard Reference Data Act (15 U.S.C. § 290e), the U.S. Secretary of Commerce may copyright SRD compilations; NIST PML provides the CODATA fundamental physical constants table (`allascii.txt`) openly to the public worldwide for unrestricted scientific, commercial, and educational use without licensing fees or usage restrictions (see NIST PML constants index at `https://physics.nist.gov/cuu/Constants/index.html` and NIST data dissemination terms at `https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications`).
- **Contents**: All 357 fundamental constants including quantities, numerical values, standard uncertainties, and units.

---

## 2. Chemical Elements: PubChem Periodic Table

- **Source**: National Center for Biotechnology Information (NCBI), National Library of Medicine (NLM), National Institutes of Health (NIH).
- **Title**: PubChem Periodic Table of Elements.
- **URL**: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON`
- **Version**: Current PubChem release (accessed October 2026).
- **Retrieval Date**: 2026-10-01.
- **License / Terms of Use**: Public Domain. NLM and NIH databases are works of the United States Government not subject to copyright in the United States (17 U.S.C. § 105). PubChem explicitly states: "PubChem is an open chemistry database at the National Institutes of Health (NIH)... PubChem data are available to the public for free, with no licensing restrictions."
- **Contents**: Elements 1 through 118 with:
  - Atomic Number (1–118)
  - Symbol
  - Element Name
  - Standard Atomic Weight (IUPAC conventional atomic mass)
  - Group (1–18)
  - Period (1–7)
  - Density (where available)
  - Melting Point (K, where available)
  - Boiling Point (K, where available)
  - Electronegativity (Pauling scale, where available)
  - First Ionization Energy (eV, where available)
  - Electron Configuration

---

## 3. Solar System & Astronomical Data

### 3.1 NASA Planetary Fact Sheet
- **Source**: National Aeronautics and Space Administration (NASA), Goddard Space Flight Center (GSFC).
- **Title**: Planetary Fact Sheet: Metric.
- **Curator**: Dr. David R. Williams.
- **URL**: `https://nssdc.gsfc.nasa.gov/planetary/factsheet/`
- **Version**: Metric Fact Sheet (current).
- **Retrieval Date**: 2026-10-01.
- **License / Terms of Use**: Public Domain. Works produced by NASA are works of the U.S. Federal Government under 17 U.S.C. § 105.
- **Contents**: 10 celestial bodies (Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto) with physical and orbital characteristics (mass, diameter, equatorial radius, density, gravity, escape velocity, rotation period, distance from Sun, perihelion, aphelion, orbital period, orbital velocity, eccentricity).

### 3.2 IAU Nominal Values
- **Source**: International Astronomical Union (IAU) 2015 Resolution B3 on Recommended Nominal Conversion Constants for Selected Solar and Planetary Properties, and IAU 2012 Resolution B2 on the re-definition of the astronomical unit.
- **URL**: `https://arxiv.org/abs/1510.07674`
- **License / Terms of Use**: Public scientific standard. Resolutions and nominal standards adopted by the International Astronomical Union General Assembly are published for open scientific use with no proprietary restrictions.
- **Standard Values**:
  - Astronomical Unit: au = 149,597,870,700 m (exact)
  - Nominal Solar Radius: R☉^N = 6.957 × 10⁸ m
  - Nominal Solar Total Irradiance: S☉^N = 1361 W/m²
  - Nominal Solar Luminosity: L☉^N = 3.828 × 10²⁶ W
  - Nominal Solar Mass Parameter: (GM)☉^N = 1.3271244 × 10²⁰ m³/s²
  - Nominal Terrestrial Equatorial Radius: ReE^N = 6.3781 × 10⁶ m
  - Nominal Terrestrial Mass Parameter: (GM)E^N = 3.986004 × 10¹⁴ m³/s²

---

## 4. Exclusion of Isotopes Dataset

- **Status**: Excluded from distribution.
- **Rationale**: Under the Standard Reference Data Act (15 U.S.C. § 290e), the U.S. Secretary of Commerce is authorized to secure copyright on behalf of the United States in Standard Reference Data (SRD) products. NIST standard reference databases (such as Database 144 / Atomic Weights and Isotopic Compositions) contain explicit reservations of copyright:
  > *"Under the Standard Reference Data Act, the U.S. Secretary of Commerce has the authority to secure copyright on Standard Reference Data products. Unless otherwise indicated, NIST SRD is copyrighted by the U.S. Secretary of Commerce on behalf of the United States."*
- Because these terms do not provide an unambiguous public-domain dedication, isotopic abundance data is excluded to strictly preserve the Apache-2.0 release licensing standards of the Reckoner Harness.

---

## 5. Precision Policy and Self-Consistency Standards

- **NASA Planetary Values**: NASA Metric Fact Sheet values are published to 3–4 significant figures (for example, Earth mass 5.97 × 10²⁴ kg, Moon mass 0.073 × 10²⁴ kg, Mars mass 0.642 × 10²⁴ kg). Where precise IAU nominal parameters are defined (such as nominal solar mass parameter (GM)☉^N = 1.3271244 × 10²⁰ m³/s² and the exact astronomical unit au = 149,597,870,700 m), those exact values are provided.
- **PubChem Abridged Atomic Weights**: PubChem publishes abridged standard atomic weights rounded to two decimal places for transition metals and certain heavy elements (for example, Iron is published as 55.84, whereas unabridged IUPAC tables cite 55.845(2)).
- **Self-Consistency Standard**: Tool observations emitted by the Go harness remain strictly self-consistent with these standard public reference datasets.

