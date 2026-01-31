# 🚀 Guide de Déploiement - NBA Betting Analyzer v5.0

## 📋 **CE QUE TU VAS AVOIR:**

✅ Dashboard "Morning Brief" avec opportunités du jour  
✅ Intégration The Odds API (~10$/mois)  
✅ Analyse automatique points + assists + rebounds  
✅ Comparaison odds entre bookmakers  
✅ Stats détaillées (Chi², outliers, R²)  
✅ Déploiement gratuit sur Render.com  

---

## 🔑 **ÉTAPE 1: The Odds API**

### 1.1 Créer un compte

1. Va sur [https://the-odds-api.com](https://the-odds-api.com)
2. Clique "Get API Key"
3. Inscris-toi (email + password)
4. Tu recevras ton **API Key** par email

### 1.2 Choisis ton plan

| Plan | Prix | Requêtes/mois | Recommandation |
|------|------|---------------|----------------|
| FREE | $0 | 500 | ✅ Pour tester |
| STARTER | $10 | 5,000 | ✅ **Idéal pour toi** |
| PRO | $50 | 50,000 | Overkill |

**Pour commencer:** FREE (500 requêtes = ~2-3 semaines d'utilisation quotidienne)

### 1.3 Note ton API Key

```
Exemple: 1a2b3c4d5e6f7g8h9i0j
```

**⚠️ IMPORTANT:** Garde cette clé secrète!

---

## 🎯 **ÉTAPE 2: Déploiement sur Render**

### 2.1 Prépare ton repo GitHub

```bash
# Clone ou update ton repo
cd Nba-betting

# Ajoute les nouveaux fichiers
git add nba_analyzer_improved.py
git add odds_api_client.py
git add dashboard_daily.html
git add requirements.txt

git commit -m "feat: v5.0 - The Odds API + Morning Brief"
git push
```

### 2.2 Configure Render

1. **Va sur [render.com](https://render.com)** et connecte-toi

2. **New Web Service**
   - Connect ton repo GitHub
   - Name: `nba-betting-analyzer`
   - Branch: `main`

3. **Build & Start:**
   ```
   Build Command: pip install -r requirements.txt
   Start Command: python nba_analyzer_improved.py
   ```

4. **Environment Variables** ⚠️ CRUCIAL:
   ```
   PORT = 10000
   DEBUG = False
   ODDS_API_KEY = [ta-clé-ici]
   ```

5. **Instance Type:**
   - Free (512 MB RAM) - Suffisant pour commencer

6. **Deploy!**
   - Clique "Create Web Service"
   - Attends 2-3 minutes le build

### 2.3 Note ton URL

Tu auras une URL type:
```
https://nba-betting-analyzer.onrender.com
```

---

## 📱 **ÉTAPE 3: Configure le Frontend**

### 3.1 Update l'URL dans dashboard_daily.html

```javascript
// Ligne ~280
const API_URL = 'https://nba-betting-analyzer.onrender.com';
```

### 3.2 Déploie sur GitHub Pages

**Option A: Via GitHub web**
1. Upload `dashboard_daily.html` sur ton repo
2. Settings → Pages → Source: main branch
3. URL sera: `https://olivierqc1.github.io/Nba-betting/dashboard_daily.html`

**Option B: Via terminal**
```bash
git add dashboard_daily.html
git commit -m "feat: morning dashboard"
git push

# GitHub Pages se met à jour automatiquement
```

---

## ☕ **ÉTAPE 4: Morning Routine**

### Ton workflow quotidien:

1. **☕ Réveille-toi, ouvre:**
   ```
   https://olivierqc1.github.io/Nba-betting/dashboard_daily.html
   ```

2. **🔍 Clique "Scanner les opportunités"**
   - Récupère les props du jour
   - Analyse avec ton modèle
   - Affiche TOP opportunités triées par edge

3. **📊 Vois les résultats:**
   - **Cards vertes = OVER** recommandés
   - **Cards rouges = UNDER** recommandés
   - Edge, Kelly%, Probabilités affichés

4. **✅ Validation rapide:**
   - Clique "Voir stats détaillées"
   - Vérifie Chi², outliers, R²
   - Si tout est OK → Place le pari

5. **⏱️ Durée totale:** 5-10 minutes max!

---

## 📊 **EXEMPLE DE MORNING BRIEF**

```
☕ Morning Brief - 31 janvier 2025, 08:30

[Filters: Edge ≥ 5%, Confiance MEDIUM+]

📊 Stats:
- Props disponibles: 87
- Props analysées: 87  
- Opportunités: 12
- Edge moyen: 8.3%

─────────────────────────────────

🟢 LeBron James vs GSW • Points
   OVER 25.5
   
   Prédiction: 28.3 pts
   Edge: +12.4%
   Kelly: 4.2%
   Confiance: HIGH
   
   📊 FanDuel: -110
   σ = 4.2 | R² = 0.81 | Chi² OK ✅
   
   [📊 Voir stats détaillées ▼]

─────────────────────────────────

🔴 Stephen Curry @ LAL • Assists
   UNDER 6.5
   
   Prédiction: 5.2 asts
   Edge: +9.1%
   Kelly: 3.1%
   Confiance: MEDIUM
   
   📊 DraftKings: -105
   σ = 1.8 | R² = 0.74 | Chi² OK ✅
   
   [📊 Voir stats détaillées ▼]

─────────────────────────────────

... 10 autres opportunités ...
```

---

## ⚙️ **CONFIGURATION AVANCÉE**

### Ajuster les filtres

Dans le dashboard:
- **Edge minimum:** 5% par défaut (conservateur), baisse à 3% si tu veux plus d'opportunités
- **Confiance:** MEDIUM+ (recommandé), change à HIGH si tu veux être ultra-sélectif

### Monitoring API Usage

```bash
# Vérifie combien de requêtes il te reste
curl https://nba-betting-analyzer.onrender.com/api/odds/usage
```

Returns:
```json
{
  "used": 42,
  "remaining": 458
}
```

**Astuce:** Tu as 500/mois FREE = ~16/jour. Le scan quotidien en utilise ~3-5.

---

## 🐛 **TROUBLESHOOTING**

### Problème 1: "Odds API non disponible"

**Cause:** API key manquante ou invalide

**Solution:**
```bash
# Sur Render, vérifie Environment Variables
ODDS_API_KEY = [ta-vraie-clé]

# Redeploy le service
```

### Problème 2: Frontend ne se connecte pas

**Cause:** URL incorrecte dans dashboard_daily.html

**Solution:**
```javascript
// Vérifie ligne ~280
const API_URL = 'https://TON-URL-RENDER.onrender.com';
```

### Problème 3: "Aucune opportunité trouvée"

**Causes possibles:**
1. Pas de matchs NBA aujourd'hui (off-season, journée sans matchs)
2. Edge minimum trop élevé
3. Tous les paris ont faible edge aujourd'hui (normal)

**Solutions:**
- Baisse l'edge minimum à 3%
- Attends les matchs du soir (props publiées vers 17h-18h)
- Regarde les props disponibles: `/api/odds/available-props`

### Problème 4: Render s'endort (plan gratuit)

**Cause:** Après 15min d'inactivité, Render met le service en veille

**Solutions:**
- Option A: Première requête du matin prend 30-60s (normal)
- Option B: Upgrade Render à $7/mois (toujours actif)
- Option C: Utilise un cron job pour "ping" le service

---

## 💰 **COÛTS TOTAUX**

| Service | Prix | Obligatoire |
|---------|------|-------------|
| The Odds API | $10/mois | ✅ Oui (ou FREE 500 req) |
| Render.com | $0 (free tier) | ✅ Oui |
| GitHub Pages | $0 | ✅ Oui |
| **TOTAL** | **$10/mois** | |

**Alternative ultra-budget:** FREE tier partout = $0/mois (limite 500 requêtes API)

---

## 🎯 **PROCHAINES AMÉLIORATIONS**

Pour optimiser le modèle et augmenter les gains:

### Phase 1: Plus de variables (court terme)
```python
# Dans nba_analyzer_improved.py, ajoute:
df['days_since_injury'] = ...
df['opponent_turnovers_forced'] = ...
df['team_offensive_rating'] = ...
df['minutes_last_3_games'] = ...
```

### Phase 2: Feature engineering (moyen terme)
- Fatigue index (back-to-backs, voyages)
- Hot/cold streaks (forme récente)
- Matchup historique spécifique
- Weather impact (outdoor games)

### Phase 3: ML avancé (long terme)
- XGBoost / Random Forest
- Feature importance analysis
- Hyperparameter tuning
- Backtesting sur saisons passées

### Phase 4: Tracking & Analytics
- Dashboard avec historique des paris
- Win rate par type de bet
- ROI cumulatif
- Bankroll management automatique

---

## 📞 **SUPPORT**

### Ressources utiles:

- **The Odds API Docs:** https://the-odds-api.com/liveapi/guides/v4/
- **Render Docs:** https://render.com/docs
- **NBA API Docs:** https://github.com/swar/nba_api

### Si ça marche pas:

1. Check les logs Render: Dashboard → Logs
2. Test l'API: `curl https://ton-url/api/health`
3. Ouvre un Issue GitHub avec les logs

---

## ✅ **CHECKLIST FINALE**

Avant de te coucher ce soir:

- [ ] Compte The Odds API créé
- [ ] API Key récupérée
- [ ] Backend déployé sur Render avec API key
- [ ] Frontend déployé sur GitHub Pages
- [ ] URL frontend mise à jour
- [ ] Test du Morning Brief: clique "Scanner"
- [ ] Bookmark le dashboard
- [ ] Configure alarme pour demain matin 🔔

**Demain matin: 5 minutes pour voir tes opportunités + placer tes paris!**

---

## 🚀 **Let's Go!**

T'es prêt pour faire de l'argent tous les matins! 💰

Questions? → GitHub Issues ou DM

Bonne chance! 🍀🏀