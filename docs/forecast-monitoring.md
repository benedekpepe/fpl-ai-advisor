# FPL Advisor – pontosság és előzetesen mentett ajánlások

## Éles ellenőrzés

A GitHub main ága a 0e5db3a javítást tartalmazza. Az éles appot a 2942747-es csapattal megnyitottuk: jelzi a korábbi szezonból/helyettesítő értékekből származó erősségeket, és már nem Neco Williams az első cserejelölt.

## Új történeti bontás

A 2024/25-ös Assistant Manager (AM) edzői sorokat kizártuk a játékosok közül. Emiatt az összesített korábbi számok változtak; a modellfájlt nem cseréltük le, a keverési súlyokat nem hangoltuk át. A jelentések külön tanított, későbbi szezonon értékelt modelleket írnak le, nem az aktuális éles modell bizonyított pontosságát.

| Tesztszezon | Összes játékos MAE | Rendszeresen játszók MAE* | Aktív játékosok percátlag-guide hibája** |
|---|---:|---:|---:|
| 2024-25 | 1.01 | 2.24 | 23.9 perc |
| 2025-26 | 0.99 | 2.37 | 24.2 perc |

*A korábbi átlagos játékidő legalább 60 perc. **Legalább 30 perc korábbi átlagos játékidő. A guide megfigyelt korábbi átlag, nem külön kalibrált percmodell.

A későbbi szezonban posztonként a becsült és tényleges pontok átlagos különbsége:
- DEF: -0.12 pont (pozitív = túlbecslés).
- FWD: +0.03 pont (pozitív = túlbecslés).
- GK: +0.02 pont (pozitív = túlbecslés).
- MID: +0.00 pont (pozitív = túlbecslés).

Ez nem indokol általános, minden védőre vagy középpályásra alkalmazott pontkorrekciót. A magas, átlagosan 92%-os szereplési valószínűségeknél a tényleges szereplés kb. 90% volt; ez enyhe túlzott bizakodást jelez, nem bizonyíték egy adott játékosra. A történeti csapaterősségi adatok későbbi módosításokat tartalmazhatnak; korabeli sérüléshíreket nem játszottunk vissza.

## Előzetes napló és visszamérés

- A mentés csak a célforduló határideje előtt készülhet; tartalmazza a játékosazonosítókat, az eredeti és ajánlott csapatot, a becsléseket, adatforrásokat és a modell/pipeline lenyomatát.
- A mentett JSON később visszatölthető az app új Keep & review a forecast részén, vagy a CLI-vel értékelhető.
- Csak teljesen lezárt és FPL-ellenőrzött forduló eredménye számít. Hiányzó eredmény nem lesz nulla.
- A mentett célheti kezdőt automatikus cserékkel, cserepadsorrenddel, vice-kapitánnyal és hitköltséggel értékeljük. Ez rögzített döntések összehasonlítása, nem a későbbi kézi módosítások visszajátszása.
- A többhetes játékoscsere-hozam külön, tulajdonolt játékosok pontkülönbségeként szerepel; nem állítjuk róla, hogy a későbbi kezdő- és kapitányválasztás teljes hatását méri.
- Letöltések/ellenőrzések közben az ajánlás megmarad a felületen; lejárt határidő után friss ajánlást kell kérni.

Az első használható GW11-mentés elkészült (2008 játékos–forduló becslés). Jelenlegi állapota: **pending**. Valós eredmények alapján még nem dönthető el, jobb volt-e a javasolt csere a megtartásnál.

## Tárolás és állapot

Alapértelmezésben a felhasználó letölti és őrzi meg a fájlt. FPL_FORECAST_LOG_DIR beállításával minden élő ajánláskéréskor automatikus, felülírás nélküli mentés készül; ehhez tartós/külsőleg csatolt könyvtár szükséges. Központi tartós felhőtárhely és határidőnkénti háttérütemező nincs aktiválva. A helyi első mentés a Gitből kizárt data/forecast_logs mappában található; a személyes mentések nem részei a repónak.

73 teszt sikeres. Az éles korábbi javítást és az új helyi Streamlit-visszamérő felületet is ellenőriztük. Az új naplózás, mérési bontások és AM-szűrés ennek a változtatásnak a része; az éles elérhetőséghez a frissített verzió telepítése szükséges.