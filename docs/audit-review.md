# FPL Advisor – commit előtti felülvizsgálat

## Eredmény

A konkrét számítási és szabálykezelési hibákat javítottuk. A forma–modell keverés alap­súlyait megtartottuk: nyolc összehasonlított változat közül egyik sem lett minden fontos hibamutatóban jobb mindkét tesztszezonon. Az éles modellfájlt nem cseréltük le.

## Javított működés

- A kapitány lehet védő vagy kapus is, ha ő a legmagasabb várható pontú kezdő.
- A −4 egyszeri csereköltséget többhetes, súlyozott pontösszeggel hasonlítjuk össze, nem heti átlaggal.
- Az ismert vételi árakból helyes eladási ár készül. Hiányos nyilvános adatoknál a költségkeret becsült voltát jelezzük.
- Free Hit után a következő fordulóra az eredeti, visszaálló keretre adunk tanácsot.
- Wildcard és Free Hit után megmarad a bankolt FT, plusz FT nélkül. A GW16-os AFCON-feltöltés csak a 2025/26-os szezonra érvényes.
- Az FT becslése opcionálisan felülírható az appban.
- A chipértékelés a menedzser költségkeretét, a kieső játékosokat, a dupla/üres heteket és a tartalék becslés heti adatait is kezeli.
- Lejáró chipet pozitív becsült előny esetén nem tartogatunk egy már nem létező későbbi lehetőségre.
- A történeti fordulók formahatásába nem kerül be a későbbi fordulók száma.
- Az élő ellenfélerősségek közvetlenül az aktuális FPL-adatokból jönnek.

## Mérés

| Tesztszezon | Modell + forma MAE | Egyszerű forma MAE | Modell + forma RMSE |
|---|---:|---:|---:|
| 2024-25 | 1.0582 | 1.1137 | 2.0892 |
| 2025-26 | 0.9932 | 1.0581 | 1.9705 |

A gyengébb/erősebb formahatás, az utolsó öt meccs formája, a minta nagyságához igazítás és a szigorúbb játékidő-korrekció is szerepelt a mérésben. A játékidő-korrekció mindkét szezonban csökkentette a MAE-t, de növelte a nagy tévedéseket jobban büntető RMSE-t. A frissebb formaátlag szintén nem adott általános javulást.

Korlátok: külön tanított összehasonlító modelleket értékeltünk, nem az éles modell pontosságát igazoltuk. A korabeli sérüléshírek nincsenek visszajátszva; a történeti ellenfélerősségek későbbi módosításokat tartalmazhatnak. A csereértékelés rögzített keretre épülő előrejelzés, nem teljes többfordulós csereütemezés.

Ellenőrzés: 56 sikeres teszt, valamint a teljes Streamlit-felület asztali és mobil ellenőrzése példaadatokkal. A publikálás a felülvizsgálattól külön lépés.

## Ellenőrzött szabályforrások

- [Hivatalos átigazolási és eladásiár-szabályok](https://www.premierleague.com/en/news/2174907/1000)
- [Hivatalos chipútmutató](https://www.premierleague.com/en/news/2174900)
- [A 2026/27-es szezon változásai](https://www.premierleague.com/en/news/4679873)