# Holdout set congelado — holdout

- **Congelado em:** 2026-05-04T16:47:34
- **Origem:** split `test`
- **Seed:** 42
- **Total de imagens:** 50
- **Motivo:** Golden set v2 - congelado para release v2.0

## Regras

1. **NAO** olhe esse split durante iteracao. So no fechamento de uma versao do modelo.
2. Se rodar `evaluate_model.py` nele mais de uma vez por release, a metrica deixa de ser
   independente — voce comeca a otimizar pra ele indiretamente.
3. Para verificar integridade:
   ```bash
   python -c "import json,hashlib,pathlib;d=json.load(open('HOLDOUT_LOCK.json'));print(all(hashlib.sha256(open(pathlib.Path('holdout/images')/f['image'],'rb').read()).hexdigest()==f['image_sha256'] for f in d['files']))"
   ```
4. Comite esse arquivo no git pra audit trail.

## Arquivos congelados

| # | Imagem | SHA256 (8 chars) |
|---|--------|------------------|
| 1 | 90_20RealBra_2_jpeg.rf.IKZcweEkEl46EayvL1hi.jpeg | `66434048` |
| 2 | 0_50RealBra_13_jpeg_jpg.rf.25qB4LJUJztmPGMxYeHe.jpg | `803b7c71` |
| 3 | 0_10RealBra_17_jpeg_jpg.rf.YibfD6Ylju7pn3WB1XKA.jpg | `968bc481` |
| 4 | image18_jpeg_jpg.rf.vCJUvNLsz7Go2EWcZaMI.jpg | `432259bc` |
| 5 | 180_2RealBra_44_jpeg_jpg.rf.jAW8OcmY9u4x8lOhMpA1.jpg | `e4469668` |
| 6 | 180_20RealBra_2_jpeg.rf.rGA441nhzjY3Gnpq4N8y.jpeg | `0e8151a8` |
| 7 | 180_10RealBra_5_jpeg.rf.BqCi4cEkhtTsZEi2QlS4.jpeg | `dadfa67e` |
| 8 | 0_5RealBra_23_jpeg_jpg.rf.zTp30pHOWKQqmPymhrzq.jpg | `b5ed4981` |
| 9 | image18_jpeg_jpg.rf.0eS0xrC7p0QM0nL7fAN0.jpg | `fcc38a50` |
| 10 | 0_2RealBra_3_jpeg_jpg.rf.ZxgLbxjK6myqGHFeZrhU.jpg | `46c1700b` |
| 11 | 90_2RealBra_8_jpeg_jpg.rf.BBjUphPZcuaUb7wyF5XK.jpg | `821b6588` |
| 12 | 62_jpg.rf.bj5VzMlNJIqRaMLYSzpl.jpg | `0d3c6f93` |
| 13 | 0_20RealBra_47_jpeg_jpg.rf.ebv9hJJBJxhEGjbp5yaU.jpg | `b734d396` |
| 14 | 90_100RealBra_29_jpeg.rf.k2MN3RxGycPMiE3OK8dU.jpeg | `58fc0959` |
| 15 | 270_20RealBra_47_jpeg.rf.h9Egq3OcZDF1hpObLSD7.jpeg | `7a9fa975` |
| 16 | 0_10RealBra_32_jpeg.rf.xlTjMZGLfV9lQYABAFvB.jpeg | `17c230be` |
| 17 | 0_10RealBra_32_jpeg.rf.wS1bWaYc0E19OhdPoWsQ.jpeg | `e43575ca` |
| 18 | 0_20RealBra_50_jpeg.rf.tw9H6mSWAuFmmNlxx8aB.jpeg | `3f4709c5` |
| 19 | 180_10RealBra_38_jpeg_jpg.rf.KPtGXSTHPClefJzNpqj9.jpg | `133a6d02` |
| 20 | 180_200RealBra_46_jpeg.rf.XiFDnFopvJHaAV4mnh3a.jpeg | `711709b5` |
| 21 | 355_jpg.rf.NIwJc03tCf5I4sfNe9kJ.jpg | `a9999aea` |
| 22 | 90_10RealBra_32_jpeg.rf.cSR6K6D7v3cCbYUHWyrF.jpeg | `32314c48` |
| 23 | 0_10RealBra_18_jpeg.rf.i08FwZanTsU63zXWMhrN.jpeg | `5640bbd9` |
| 24 | 7a62ff8f-200REAIS-IMG20_jpg.rf.rIqz5oZmchh98fnwlHHx.jpg | `3ef90157` |
| 25 | 180_100RealBra_4_jpeg.rf.dDxEoxur1WOEaXRGnczW.jpeg | `af616065` |
| 26 | a97fe423-20REAIS-IMG18_jpg.rf.pQl5VCbMJgsIwkdfFgZs.jpg | `b1b140bb` |
| 27 | 90_20RealBra_47_jpeg.rf.OayJKmr06BcXWnm5kJd6.jpeg | `90c671f0` |
| 28 | 90_50RealBra_4_jpeg_jpg.rf.sVeEHCfDTyUbSjwzwsBz.jpg | `67dc29a0` |
| 29 | 270_20RealBra_46_jpeg_jpg.rf.L7zw2Nrro8uK0M5nPwjY.jpg | `e3e862f9` |
| 30 | 180_10RealBra_38_jpeg_jpg.rf.LgKVVkkZwx6Wv7Cx6GEP.jpg | `c34818f4` |
| 31 | 270_2RealBra_8_jpeg_jpg.rf.3CPgoX5U3tEFGeCx7M3f.jpg | `a04d4f1d` |
| 32 | 90_100RealBra_27_jpeg_jpg.rf.zbra6w7yMjsxeJ4XPkns.jpg | `09415525` |
| 33 | 180_2RealBra_8_jpeg_jpg.rf.JHoggDAX3t8buPrhyz6M.jpg | `ef0abf0d` |
| 34 | 0_100RealBra_21_jpeg_jpg.rf.memABXG2J1Q4yhBpcvLM.jpg | `1dd8626d` |
| 35 | 125_jpg.rf.9YyPYaZVbnpvxK5tfPk0.jpg | `73464dae` |
| 36 | 90_50RealBra_4_jpeg_jpg.rf.l40dONS6ZCNpjxTFUWxq.jpg | `93ec46df` |
| 37 | 20240925_195113_jpg.rf.QN2ZBetTyu7nm7iHpI2F.jpg | `f0df29ee` |
| 38 | 10reais_4_jpeg.rf.3Q9eYkv9RmkfMkBREw8K.jpeg | `e6dea721` |
| 39 | 180_10RealBra_35_jpeg_jpg.rf.DogSZkf60w0qe00qYPpZ.jpg | `3dd15f4e` |
| 40 | 20240925_193457_jpg.rf.PIV5PpUlB7w9wGLi2be8.jpg | `c995a8ee` |
| 41 | 270_10RealBra_32_jpeg.rf.4jTjykwUzJBpcCxEFClq.jpeg | `6800ff34` |
| 42 | 0_2RealBra_2_jpeg.rf.Pg6MtxdD80MY2hoU3qWH.jpeg | `c85a8b60` |
| 43 | 270_100RealBra_21_jpeg_jpg.rf.NhxdUVHZKSHeWsgPcN4x.jpg | `d2ad4df3` |
| 44 | 220_jpg.rf.HSI49cBXryAKTTJJovmt.jpg | `e6f49d56` |
| 45 | 90_10RealBra_32_jpeg.rf.J23KOtM5QbdTaszjPoWj.jpeg | `d0fc576a` |
| 46 | 180_2RealBra_2_jpeg.rf.cKrbm8tZGdvDQmvdpb26.jpeg | `c393ecde` |
| 47 | 0_10RealBra_5_jpeg.rf.xdoSUYHqUThmgc72sGyk.jpeg | `af54187c` |
| 48 | cedulas_39_brightness_aug_0-6_png.rf.a37wLzZ8A3Qb2rz66MfC.png | `8c657d78` |
| 49 | 270_50RealBra_2_jpeg.rf.vIaYsKJj8bSsnx4sPJOj.jpeg | `2f648222` |
| 50 | 56-100_jpg.rf.J1QOoZfJaAgCc4qcxHkV.jpg | `483ec2b0` |