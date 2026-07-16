"""Visual theme for the Streamlit frontend: a blue-gradient "glass" medical
icon set (original artwork, not traced from any third-party icon pack) plus
the CSS that gives the app a cohesive, professional look.
"""

import itertools

COLORS = {
    "light": "#5CC8F2",
    "primary": "#2196F3",
    "dark": "#0D47A1",
    "bg": "#F3F8FD",
    "card": "#FFFFFF",
    "text": "#0F2942",
    "muted": "#5A7184",
    "danger": "#E64545",
    "danger_soft": "#FDECEC",
    "success": "#1FA971",
    "success_soft": "#E9F9F1",
}

_uid_counter = itertools.count()


def _uid() -> str:
    return f"ic{next(_uid_counter)}"


def _grad(uid: str) -> str:
    return (
        f'<linearGradient id="{uid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0%" stop-color="{COLORS["light"]}"/>'
        f'<stop offset="100%" stop-color="{COLORS["dark"]}"/>'
        f"</linearGradient>"
    )


def _svg(size: int, uid: str, body: str) -> str:
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 64 64" '
        f'xmlns="http://www.w3.org/2000/svg"><defs>{_grad(uid)}</defs>{body}</svg>'
    )


def icon_brand(size: int = 40) -> str:
    """Rounded badge with a medical cross + pulse line -- the app's mark."""
    uid = _uid()
    body = f"""
    <rect x="4" y="4" width="56" height="56" rx="16" fill="url(#{uid})"/>
    <path d="M18 33 L26 33 L29 24 L34 42 L38 33 L46 33" fill="none"
          stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9"/>
    <rect x="28" y="14" width="8" height="20" rx="2" fill="white" opacity="0.95"/>
    <rect x="20" y="22" width="24" height="8" rx="2" fill="white" opacity="0.95"/>
    """
    return _svg(size, uid, body)


def icon_syringe(size: int = 32) -> str:
    uid = _uid()
    body = f"""
    <g transform="rotate(45 32 32)">
      <rect x="22" y="18" width="20" height="26" rx="3" fill="url(#{uid})"/>
      <rect x="26" y="10" width="12" height="10" rx="2" fill="url(#{uid})"/>
      <rect x="29" y="4" width="6" height="8" fill="url(#{uid})"/>
      <rect x="24" y="44" width="16" height="5" rx="1.5" fill="url(#{uid})"/>
      <line x1="32" y1="49" x2="32" y2="58" stroke="url(#{uid})" stroke-width="3" stroke-linecap="round"/>
      <rect x="25" y="24" width="14" height="3" rx="1.5" fill="white" opacity="0.55"/>
      <rect x="25" y="30" width="14" height="3" rx="1.5" fill="white" opacity="0.4"/>
    </g>
    """
    return _svg(size, uid, body)


def icon_capsule(size: int = 32) -> str:
    uid = _uid()
    body = f"""
    <g transform="rotate(-35 32 32)">
      <rect x="12" y="24" width="40" height="16" rx="8" fill="url(#{uid})"/>
      <path d="M32 24 H44 A8 8 0 0 1 44 40 H32 Z" fill="white" opacity="0.85"/>
    </g>
    """
    return _svg(size, uid, body)


def icon_clipboard(size: int = 32) -> str:
    uid = _uid()
    body = f"""
    <rect x="14" y="10" width="36" height="46" rx="5" fill="url(#{uid})"/>
    <rect x="24" y="6" width="16" height="9" rx="3" fill="{COLORS['dark']}"/>
    <polyline points="21,34 27,34 30,27 35,42 39,34 44,34" fill="none"
              stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    """
    return _svg(size, uid, body)


def icon_folder(size: int = 32) -> str:
    uid = _uid()
    body = f"""
    <path d="M10 20 a4 4 0 0 1 4-4 h12 l5 6 h19 a4 4 0 0 1 4 4 v22 a4 4 0 0 1-4 4
             H14 a4 4 0 0 1-4-4 Z" fill="url(#{uid})"/>
    <circle cx="32" cy="34" r="9" fill="white" opacity="0.9"/>
    <rect x="29" y="29" width="6" height="10" rx="1.5" fill="{COLORS['dark']}"/>
    <rect x="27" y="31" width="10" height="6" rx="1.5" fill="{COLORS['dark']}"/>
    """
    return _svg(size, uid, body)


def icon_clock(size: int = 28) -> str:
    uid = _uid()
    body = f"""
    <circle cx="32" cy="32" r="24" fill="url(#{uid})"/>
    <line x1="32" y1="32" x2="32" y2="18" stroke="white" stroke-width="3" stroke-linecap="round"/>
    <line x1="32" y1="32" x2="42" y2="36" stroke="white" stroke-width="3" stroke-linecap="round"/>
    """
    return _svg(size, uid, body)


def icon_test_tube(size: int = 28) -> str:
    uid = _uid()
    body = f"""
    <g transform="rotate(15 32 32)">
      <path d="M26 10 h12 v30 a6 6 0 0 1-12 0 Z" fill="url(#{uid})"/>
      <path d="M26 30 h12 v10 a6 6 0 0 1-12 0 Z" fill="white" opacity="0.55"/>
      <rect x="24" y="8" width="16" height="5" rx="2" fill="{COLORS['dark']}"/>
    </g>
    """
    return _svg(size, uid, body)


def icon_alert(size: int = 30) -> str:
    """Siren-style alert icon (positive / risk result)."""
    uid = _uid()
    body = f"""
    <path d="M18 44 a14 14 0 0 1 28 0 Z" fill="url(#{uid})"/>
    <rect x="14" y="44" width="36" height="6" rx="3" fill="{COLORS['dark']}"/>
    <line x1="32" y1="12" x2="32" y2="18" stroke="url(#{uid})" stroke-width="3" stroke-linecap="round"/>
    <line x1="18" y1="18" x2="22" y2="22" stroke="url(#{uid})" stroke-width="3" stroke-linecap="round"/>
    <line x1="46" y1="18" x2="42" y2="22" stroke="url(#{uid})" stroke-width="3" stroke-linecap="round"/>
    """
    return _svg(size, uid, body)


def icon_check(size: int = 30) -> str:
    """Check-in-circle icon (clear / no-risk result)."""
    uid = _uid()
    body = f"""
    <circle cx="32" cy="32" r="24" fill="url(#{uid})"/>
    <polyline points="21,33 28,40 44,24" fill="none" stroke="white"
              stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    """
    return _svg(size, uid, body)


CSS = f"""
<style>
:root {{
  --primary-light: {COLORS['light']};
  --primary: {COLORS['primary']};
  --primary-dark: {COLORS['dark']};
}}

[data-testid="stAppViewContainer"] {{
  background: {COLORS['bg']};
}}
.block-container {{
  padding-top: 1.5rem;
  max-width: 1100px;
}}
h1, h2, h3, h4, p, span, label, div {{
  color: {COLORS['text']};
}}

.med-hero {{
  display: flex; align-items: center; gap: 18px;
  background: linear-gradient(135deg, {COLORS['light']} 0%, {COLORS['primary']} 55%, {COLORS['dark']} 100%);
  border-radius: 20px; padding: 22px 28px; margin-bottom: 22px;
  box-shadow: 0 10px 30px rgba(13, 71, 161, 0.25);
}}
.med-hero h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
.med-hero p {{ color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 0.92rem; }}

.med-card {{
  background: {COLORS['card']}; border-radius: 16px; padding: 20px 22px;
  border: 1px solid rgba(33, 150, 243, 0.10);
  box-shadow: 0 4px 18px rgba(15, 41, 66, 0.06);
  margin-bottom: 16px;
}}
.med-card h3 {{ display: flex; align-items: center; gap: 10px; margin-top: 0; font-size: 1.05rem; }}

.stat-card {{
  display: flex; align-items: center; gap: 14px;
  background: {COLORS['card']}; border-radius: 14px; padding: 14px 16px;
  border: 1px solid rgba(33, 150, 243, 0.10);
  box-shadow: 0 4px 14px rgba(15, 41, 66, 0.05);
}}
.stat-card .stat-value {{ font-size: 1.35rem; font-weight: 700; color: {COLORS['dark']}; line-height: 1.1; }}
.stat-card .stat-label {{ font-size: 0.8rem; color: {COLORS['muted']}; }}

.alert-banner {{
  display: flex; align-items: center; gap: 16px;
  border-radius: 16px; padding: 18px 22px; margin: 10px 0;
}}
.alert-banner.risk {{ background: {COLORS['danger_soft']}; border: 1px solid rgba(230,69,69,0.25); }}
.alert-banner.clear {{ background: {COLORS['success_soft']}; border: 1px solid rgba(31,169,113,0.25); }}
.alert-banner .alert-title {{ font-size: 1.1rem; font-weight: 700; margin: 0; }}
.alert-banner.risk .alert-title {{ color: {COLORS['danger']}; }}
.alert-banner.clear .alert-title {{ color: {COLORS['success']}; }}
.alert-banner .alert-sub {{ font-size: 0.85rem; color: {COLORS['muted']}; margin: 2px 0 0 0; }}

[data-testid="stDataFrame"] {{
  border-radius: 12px; overflow: hidden;
  box-shadow: 0 4px 14px rgba(15, 41, 66, 0.06);
}}

div[data-testid="stForm"] {{
  background: {COLORS['card']}; border-radius: 16px; padding: 22px;
  border: 1px solid rgba(33, 150, 243, 0.10);
  box-shadow: 0 4px 18px rgba(15, 41, 66, 0.06);
}}

button[kind="primary"], [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['dark']}) !important;
  border: none !important; box-shadow: 0 4px 12px rgba(33,150,243,0.35) !important;
}}
</style>
"""
