# Konsa–Calafiori: a pontbecslés ellenőrzése

Ellenőrzés: 2026-11-08. Cél: GW11; forma: csak a lezárt és FPL-ellenőrzött GW1–9.

## Az aktuális különbség

A hivatalos FPL-adatokban mindketten az Arsenal védői, azonos következő ellenfelekkel és teljes elérhetőséggel. Nincs játékosazonosító- vagy ellenfélcsere.

| Adat | Konsa | Calafiori |
| --- | ---: | ---: |
| Élő FPL points per match | 6,3 | 5,7 |
| Lezárt fordulók összpontja | 49 | 51 |
| Lezárt fordulók, pályára lépéssel | 8 | 9 |
| Lezárt fordulók pontátlaga, a kimaradást is számolva | 5,44 | 5,67 |
| Utolsó 5 lezárt mérkőzés: percátlag | 90 | 74,6 |
| Utolsó 5 lezárt mérkőzés: pontátlag | 7,6 | 4,6 |
| Utolsó 5 lezárt mérkőzés: xGI-átlag | 0,126 | 0,206 |
| Utolsó 5 lezárt mérkőzés: védekezési hozzájárulásból kapott pont | 4 | 0 |
| Régi pontmodell: pályára lépés esélye | 93,49% | 87,04% |
| Régi pontmodell: pont, feltéve hogy játszik | 2,65 | 4,07 |
| Ezek szorzata, a formakeverés előtt | 2,48 | 3,54 |
| Következő fordulóra mutatott becslés | 3,37 | 4,18 |

A headline FPL-átlag a pályára lépéseket számolja, és már tartalmazhatja a még nem ellenőrzött GW10 eredményét. Az app a GW10-et következetesen kihagyta. A formához használt szezonátlag a teljesen kimaradt meccset is beleszámolja: Konsánál az első forduló nulla ponttal szerepel. Ez önmagában csak kis részét magyarázza a különbségnek.

A pontos képlet ezen a fordulón:

- Konsa: 70% × 2,4777 + 30% × 5,4444 = 3,3678.
- Calafiori: 70% × 3,5392 + 30% × 5,6667 = 4,1775.

A különbség döntő része a regressziós modellből ered; nem a játékidő-esélyből, amely Konsának kedvez. A modell többek között árat, xGI-t, BPS-t és korábbi pontokat használ. Az egzakt fa-hozzájárulásokban a korábbi pontok, BPS és az ár is Calafiori relatív előnyéhez járultak hozzá. Ezek tanult összefüggések, nem bizonyíték arra, hogy ő lesz a jobb választás.

## Talált hiányosság és javítás

Az élő betöltés nem őrizte meg a defensive_contribution, clearances_blocks_interceptions, recoveries és tackles adatokat. Ezek bekerültek. A feature-előkészítés külön, kizárólag korábbi mérkőzésekből számol védekezési kontextust. A játékos részleteinél megjelenik az utolsó öt meccsen kapott védekezési bónusz átlaga.

A jelenlegi pontmodellnek nincs külön defensive contribution bemenete. A bónusz ettől még benne van a korábbi összpontszámokban, ezért azt nem szabad egyszerűen még egyszer hozzáadni. A korábbi, azonosítható modellfrissítés commitja d72f1c0, amelynek leírása 2025-26-os tanításra utal; az artifact nem tartalmaz részletes tanítási metaadatokat. Nem állítjuk róla, hogy egyáltalán nem találkozott az új szabály szerinti pontokkal.

A szabály szerint a védők 10, a középpályások/csatárok 12 megfelelő védekezési hozzájárulásért kapnak legfeljebb 2 pontot mérkőzésenként. Hiányzó adat nem lett nulla, régebbi szezonokra nem számolunk visszamenőleg új bónuszt.

## Kipróbált új modell — nincs bekapcsolva

A bővített modell a clean sheet, xGC, védekezési hozzájárulás és a belőle kapott pontok megelőző 3/5 meccses átlagát is használta. Ugyanolyan paraméterekkel, azonos tanítási adatokon, azonos formakeveréssel hasonlítottuk össze az eredeti bemenetű újratanítással. A teljes pontszám volt a cél; nem adtunk hozzá még egyszer védekezési bónuszt.

| Ellenőrzés | Eredeti bemenetekkel újratanított MAE | Bővített modell MAE |
| --- | ---: | ---: |
| 2025-26 GW25–38, összes játékos | 0,9551 | 0,9652 |
| Ugyanez, rendszeresen játszó védők | 2,2209 | 2,2525 |
| 2026-27 lezárt GW2–9, összes játékos | 1,1658 | 1,1665 |
| Ugyanez, rendszeresen játszó védők | 2,3629 | 2,3837 |

Az aktív csoport korábbi átlagos játékideje legalább 30 perc. A történeti teszt előtt a tanítás csak a 2025-26-os GW24-ig tartott, korábbi szezonokkal kiegészítve. Az élő teszt előtt a tanítás kizárólag 2025-26-ig tartott. Az ellenfelek történeti erősségértékei későbbi módosításokat tartalmazhatnak, sérüléshíreket nem játszottunk vissza. Ezeket a tesztidőszakokat már megvizsgáltuk; nem szabad érintetlen hangolási mintaként kezelni őket.

A jelölt a két vizsgált játékosnál Konsa 4,34 / Calafiori 4,07 becslést adott, de a szélesebb történeti teszten romlott a pontossága. Ezért a pontartifact és a súlyozás változatlan. Nincs Konsára, Arsenal-védőkre vagy a rangsorban elfoglalt helyre szabott pontemelés.

Újrafuttatás (élő adatok változhatnak):

```sh
FPL_DATA_SOURCE=live python -m src.model.defence_audit --cache-dir work/validation --output docs/defence_context_validation.json --compare Konsa Calafiori
```

PowerShellben előbb: `$env:FPL_DATA_SOURCE="live"`, majd a `python -m ...` parancs. A parancs nem cseréli le a modellfájlt.

A számszerű gépi jelentés: [defence_context_validation.json](defence_context_validation.json).

Források: [FPL bootstrap](https://fantasy.premierleague.com/api/bootstrap-static/), [Konsa története](https://fantasy.premierleague.com/api/element-summary/31/), [Calafiori története](https://fantasy.premierleague.com/api/element-summary/8/), [hivatalos védekezési pontozás](https://www.premierleague.com/en/news/4361991/).
