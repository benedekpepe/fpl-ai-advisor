# Neco Williams–Gvardiol: hibavizsgálat

A 2026-11-08-i adatpillanatban az élő API mind a 20 klubnál nullás támadó- és védőerősségi mezőket adott. Ezek még nem inicializált értékek, nem valódi ellenfélerősségek. A modell ezeket mégis érvényes számként kapta meg, így túlzott pontbecslések készültek. Ez az adatminőségi ellenőrzés az előző javításból kimaradt.

| GW11, sérüléssel korrigálva | Hibás bemenettel | Javított bemenettel |
|---|---:|---:|
| Neco Williams | 5,79 | 3,21 |
| Gvardiol | 3,65 | 2,92 |

A javítás érvényes élő erősségeket használ, ha vannak. Nulla/hiányzó mezőnél az utolsó lezárt szezon adatai következnek, klubkód alapján párosítva, nem változó csapatazonosítóval. Ismeretlen klubnál a valid értékek mediánja ad semleges becslést. Ha semmilyen védhető kiinduló adat nincs, nem számolunk nulla erősséggel. Mindkét pontbecslési ág ugyanazt a feloldást használja, a felület jelzi a helyettesítő értékeket.

Az első 9 lezárt fordulóban Williams pontátlaga 3,56; Gvardiolé 4,89. Az utolsó öt megfigyelt meccs átlagos játékideje ugyanakkor 88,6, illetve 62,2 perc. Gvardiol élő státusza kétes, 75%-os játékeséllyel. Ezek indokolják, hogy a jövőbeli becslés ne egyszerűen a múltbeli összpontszámot kövesse; az 5,79-es Williams-becslést azonban a hibás ellenfélerősség jelentősen megemelte.

A teljes 2942747-es keretet újraszámoltuk. Az adott pillanatban a javított csereoptimalizálás már nem Williamst, hanem Tarkowskit adta első jelöltnek, +1,40 súlyozott összpontelőnnyel a következő négy fordulóra. Ez a modell adott pillanatbeli eredménye, nem biztos kimenetel vagy önálló vételi felszólítás.

62 teszt sikeres, köztük az érvénytelen erősségek, az eltérő szezonbeli csapatazonosítók, a feljutó klubok és mindkét becslési ág regressziós tesztjei. A forma–modell keverés súlyait nem módosítottuk. A leírás a javítás ellenőrzésekor készült; az éles app frissítése külön telepítési lépés.

Források: [FPL bootstrap](https://fantasy.premierleague.com/api/bootstrap-static/), [Williams meccsadatai](https://fantasy.premierleague.com/api/element-summary/469/), [Gvardiol meccsadatai](https://fantasy.premierleague.com/api/element-summary/391/), valamint a projektben használt vaastav 2025-26 csapaterősségi adatok. A prior szezon erősségei becslések: nem állítjuk, hogy az idei erőviszonyokat pontosan leírják.
