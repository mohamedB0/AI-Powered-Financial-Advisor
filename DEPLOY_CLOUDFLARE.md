# Deploy behind Cloudflare Tunnel

Makes the AI Financial Advisor reachable by anyone at `https://yourdomain.com`.
HTTPS is terminated at the Cloudflare edge — no open ports, no public IP needed.

## Architecture

```
User → https://yourdomain.com → Cloudflare edge (TLS)
      → cloudflared tunnel     → caddy:80 (internal router)
                                ├── /api/*        → ui:5000
                                ├── /auth /query /conversations /ingest
                                │   /stocks /bvc /pdf /speech /vision → api:8000
                                └── everything else → ui:5000
```

## Prerequisites

- A domain whose **DNS is managed by Cloudflare** (free plan is fine).
- A server with Docker + Docker Compose v2 and **≥ 4GB RAM** (Ubuntu/Debian).
  Recommended ~$6/mo: Hetzner CX22, DigitalOcean basic, etc.

## 1. Copy the project to the server

```bash
git clone <your-repo> /opt/financial-advisor && cd /opt/financial-advisor
```

`.env` and the tunnel credentials are secrets — never commit them. Copy `.env`
to the server via scp/SFTP and fill in the production values (step 4).

## 2. Create the Cloudflare Tunnel

Install `cloudflared` on the server once (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):

```bash
cloudflared tunnel login            # opens a browser; authorize your domain
cloudflared tunnel create financial-advisor   # → prints TUNNEL_ID
```

Copy the generated credential file into the repo (gitignored):

```bash
mkdir -p deploy/cloudflared
cp ~/.cloudflared/<TUNNEL_ID>.json deploy/cloudflared/
```

Edit `deploy/cloudflared/config.yml`:
- replace `YOUR_TUNNEL_ID` with your actual tunnel ID
- replace `yourdomain.com` with your domain

Point DNS at the tunnel:

```bash
cloudflared tunnel route dns financial-advisor yourdomain.com
```

## 3. (Optional) Configure the Caddy origin

`deploy/caddy/Caddyfile` binds `:80` internally and path-routes by prefix — it does
**not** contain your domain, so no edits are needed. Only touch it if you change
how paths map to the UI vs API.

## 4. Production `.env` on the server

Required changes vs dev:

```ini
ENVIRONMENT=production
AUTH_COOKIE_SECURE=true          # HTTPS → cookie auth works
CORS_ORIGINS=https://yourdomain.com
REDIS_PASSWORD=<strong-random>    # must match what the API/UI/workers read
POSTGRES_PASSWORD=<strong-random>
FLOWER_PASSWORD=<strong-random>
JWT_SECRET=<keep-your-strong-secret>
GOOGLE_CLIENT_ID=<unchanged>
```

## 5. Allow Google Sign-In for the public origin

Google Cloud Console → APIs & Services → Credentials → your OAuth client
(Web application) → **Authorized JavaScript origins** → add:

```
https://yourdomain.com
```

GSI is origin-based (no callback URL needed). Without this, the Google button
renders but logs you out immediately / stays blank.

## 6. Start the stack

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

First boot only — apply DB migrations:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec api alembic upgrade head
```

## 7. Verify

- `curl -sI https://yourdomain.com` → 200
- Open it in a browser → sign in (Google or email/password)
- `docker compose -f docker-compose.yml -f docker-compose.prod.yml ps`
- `docker compose -f docker-compose.yml -f docker-compose.prod.yml logs cloudflared caddy`

## Security checklist

- [ ] Firewall: only SSH (22) open; everything else via the tunnel.
- [ ] `REDIS_PASSWORD`, `POSTGRES_PASSWORD`, `FLOWER_PASSWORD` rotated to strong values.
- [ ] `deploy/cloudflared/*.json` and `.env` never committed.
- [ ] `AUTH_COOKIE_SECURE=true` (set in docker-compose.prod.yml — do not remove).
- [ ] Optional: add Cloudflare **Access** (Zero Trust → Access → Applications) with
      email OTP so only invited people can open the app at all.

## Troubleshooting

- **Tunnel doesn't come up**: `cloudflared tunnel info financial-advisor` and check
  the JSON filename matches `credentials-file` in config.yml.
- **Google button empty**: origin not in Authorized JavaScript origins (step 5).
- **Login succeeds then logs out instantly**: `AUTH_COOKIE_SECURE` mismatch —
  must be `true` over HTTPS.
- **Ports exposed**: `ss -ltn` — only `:22` and `127.0.0.1:*` should listen.
