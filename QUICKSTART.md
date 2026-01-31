# ⚡ QUICKSTART - NBA Betting Analyzer v5.0

## 🎯 **TON SYSTÈME AUTOMATIQUE EST PRÊT!**

### Ce que tu as maintenant:

✅ **Backend Python** avec:
- The Odds API (~10$/mois) pour odds en temps réel
- Analyse automatique points/assists/rebounds
- Test Chi-carré, détection outliers, régression
- Endpoint `/api/daily-opportunities` qui fait TOUT le travail

✅ **Dashboard "Morning Brief"** qui affiche:
- TOP opportunités du jour (edge ≥ 5%)
- Comparaison odds entre bookmakers
- Stats détaillées (Chi², R², outliers)
- Recommandations OVER/UNDER/SKIP
- Kelly criterion pour sizing

---

## 🚀 **DÉPLOIEMENT EN 3 ÉTAPES**

### 1️⃣ The Odds API (2 minutes)

```bash
# 1. Va sur https://the-odds-api.com
# 2. Clique "Get API Key"
# 3. Inscris-toi (email + password)
# 4. Copie ton API Key

# Exemple: 1a2b3c4d5e6f7g8h9i0j
```

**Plan recommandé:** FREE pour tester (500 requêtes/mois)  
Upgrade à $10/mois après si ça marche bien.

---

### 2️⃣ Déploiement Render (5 minutes)

Sur Render.com (déjà connecté):

```
New Web Service
├─ Repo: ton-github/Nba-betting
├─ Branch: main
├─ Build: pip install -r requirements.txt
├─ Start: python nba_analyzer_improved.py
└─ Environment Variables:
   ├─ PORT = 10000
   ├─ DEBUG = False
   └─ ODDS_API_KEY = [ta-clé-ici] ⚠️ IMPORTANT
```

Clique Deploy → Attends 2-3 min → URL prête!

---

### 3️⃣ Frontend GitHub Pages (2 minutes)

```bash
# 1. Upload les fichiers sur GitHub
git add dashboard_daily.html odds_api_client.py nba_analyzer_improved.py
git commit -m "feat: v5 - The Odds API + Morning Brief"
git push

# 2. Update l'URL dans dashboard_daily.html ligne ~280
const API_URL = 'https://ton-app.onrender.com';

# 3. Repush
git add dashboard_daily.html
git commit -m "fix: update API URL"
git push
```

**Ton URL finale:**
```
https://olivierqc1.github.io/Nba-betting/dashboard_daily.html
```

---

## ☕ **MORNING ROUTINE (5 minutes)**

### Tous les matins:

1. **Ouvre le dashboard**
   ```
   https://olivierqc1.github.io/Nba-betting/dashboard_daily.html
   ```

2. **Clique "Scanner les opportunités"**
   - Le système récupère les props du jour via The Odds API
   - Analyse chaque prop avec ton modèle ML
   - Affiche uniquement edge ≥ 5%

3. **Vois les résultats** (exemple):
   ```
   🟢 LeBron James OVER 25.5 pts
      Edge: +12.4% | Kelly: 4.2% | HIGH confidence
      FanDuel -110 | R²=0.81 | Chi² OK ✅
   
   🔴 Curry UNDER 6.5 asts  
      Edge: +9.1% | Kelly: 3.1% | MEDIUM confidence
      DraftKings -105 | R²=0.74 | Chi² OK ✅
   ```

4. **Valide rapidement:**
   - Clique "Voir stats détaillées"
   - Check Chi², outliers, splits
   - Si OK → Place le pari!

5. **Done!** Retourne à ton café ☕

---

## 📊 **FICHIERS CRÉÉS**

```
Nba-betting/
├─ nba_analyzer_improved.py          ← Backend principal
├─ odds_api_client.py                ← Client The Odds API
├─ dashboard_daily.html              ← Morning Brief UI
├─ index_v4.html                     ← Analyse manuelle (backup)
├─ requirements.txt                  ← Dépendances Python
├─ .env.example                      ← Template config
├─ DEPLOYMENT_GUIDE.md               ← Guide détaillé
└─ QUICKSTART.md (ce fichier)        ← Guide rapide
```

---

## 🎯 **VARIABLES À CONFIGURER**

### Sur Render (Environment Variables):

| Variable | Valeur | Obligatoire |
|----------|--------|-------------|
| `PORT` | 10000 | ✅ Oui |
| `DEBUG` | False | ✅ Oui |
| `ODDS_API_KEY` | ta-clé | ✅ **OUI!** |

### Dans dashboard_daily.html:

```javascript
// Ligne ~280
const API_URL = 'https://ton-app-render.onrender.com';
```

---

## 🔥 **ENDPOINTS DISPONIBLES**

Ton backend expose:

```bash
# Health check
GET /api/health

# Daily scan (PRINCIPAL)
GET /api/daily-opportunities?min_edge=5&min_confidence=MEDIUM

# Analyse manuelle
POST /api/analyze
{
  "player": "LeBron James",
  "opponent": "GSW",
  "is_home": true,
  "stat_type": "points",
  "line": 25.5,
  "remove_outliers": true
}

# Analyse 3 stats en un coup
POST /api/analyze-all

# Props disponibles (sans analyse)
GET /api/odds/available-props

# Usage API
GET /api/odds/usage
```

---

## 💰 **COÛTS**

| Service | Prix |
|---------|------|
| The Odds API (FREE) | $0/mois |
| Render.com (FREE) | $0/mois |
| GitHub Pages | $0/mois |
| **TOTAL** | **$0/mois** |

Upgrade The Odds API à $10/mois quand tu dépasses 500 requêtes.

---

## ✅ **CHECKLIST**

Avant de dormir ce soir:

- [ ] API Key The Odds API récupérée
- [ ] Backend déployé sur Render avec API key
- [ ] Frontend sur GitHub Pages
- [ ] URL mise à jour dans dashboard
- [ ] Test du scan: clique "Scanner"
- [ ] Bookmark le dashboard
- [ ] Alarme 8h demain matin

**Demain: 5 min pour voir tes opportunités!**

---

## 🐛 **SI ÇA MARCHE PAS**

### Backend ne démarre pas:
```bash
# Check les logs Render
Dashboard → Logs → Cherche "ERROR"

# Vérifie l'API key
curl https://ton-url/api/health
```

### Frontend erreur 404:
```bash
# Vérifie que GitHub Pages est activé
Repo → Settings → Pages → Source: main

# URL correcte?
https://olivierqc1.github.io/Nba-betting/dashboard_daily.html
```

### "Odds API non disponible":
```bash
# Check Environment Variable sur Render
ODDS_API_KEY = [ta-vraie-clé-sans-brackets]

# Redeploy après changement
```

---

## 🚀 **C'EST TOUT!**

T'es **prêt** pour faire de l'argent tous les matins! 💰

Le système fait:
- ✅ Récupération odds
- ✅ Analyse automatique
- ✅ Filtrage edge ≥ 5%
- ✅ Calcul Kelly
- ✅ Validation Chi²

**Toi tu fais:**
- ☕ Café
- 🖱️ Clic "Scanner"
- 👀 Regarde les opportunités
- 💸 Place les paris

**Temps total: 5 minutes!**

---

## 📈 **PROCHAINE ÉTAPE (après 1 semaine)**

Une fois que tu vois que ça marche:

1. **Track tes résultats** (Excel simple)
   - Date, Joueur, Pari, Edge%, Résultat
   
2. **Ajuste les filtres**
   - Si trop d'opportunités → monte à edge ≥ 7%
   - Si pas assez → baisse à edge ≥ 3%

3. **Améliore le modèle**
   - Ajoute variables (fatigue, matchup history)
   - Test d'autres algos (XGBoost)
   - Backtest sur saisons passées

4. **Scale up**
   - Upgrade The Odds API ($10/mois)
   - Track ROI automatiquement
   - Dashboard avec graphs

---

## 💪 **LET'S FUCKING GO!**

Questions? → GitHub Issues

Bonne chance champion! 🏀💰🚀