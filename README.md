# SIR (Smart Incident Radar)

SIR is a monorepo for a city incident dashboard that visualizes **recent incidents (last 24h)** and **future risk predictions** on an interactive map (**Riyadh MVP**).

Monorepo packages:
- `sir-ai` — AI / prediction engine (placeholder / future)
- `sir-client-dashboard` — React + Vite + MapLibre dashboard
- `sir-data-base` — PostgreSQL + PostGIS (migrations + seed data)
- `sir-server` — NestJS REST API

---

## Demo

A short demo video / screenshots are available here:
- https://drive.google.com/drive/folders/1iEcrKjh25Yd-ep3JpGMREvSRL1O15nag?usp=drive_link

---

## Prerequisites

Make sure you have the following installed:

- Node.js (LTS recommended)
- npm
- Docker Desktop
- Git

Notes:
- Frontend runs with **Vite**
- Backend is built with **NestJS**
- Database uses **PostGIS** for geospatial queries
