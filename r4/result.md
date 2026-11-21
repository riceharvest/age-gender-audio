PASS: age-gender-audio 0.1.0 R4 gates, frozen d18b575, n=1200 CV17-en, list sha256 90d1c69b...7fe07.

- incumbent: gender acc 0.9492, age MAE 9.741 (10/1200 top-class child)
- fp32: gender 0.9600 / MAE 9.177; diff +1.08 pp CI [+0.00,+2.17] (need >= -2.0); age -0.565 yr CI [-1.113,+0.018] (need <= +1.0). PASS
- int8: gender 0.9583 / MAE 9.192; diff +0.92 pp CI [-0.17,+2.00]; age -0.550 yr CI [-1.096,+0.024]. PASS
- parity int8-fp32: gender -0.17 pp (0.5), age +0.015 yr (0.3). PASS
- pytest: 18 passed. Raw: r4/gate_r4.json. Wording: within the pre-registered margin; not claimed to beat the incumbent.
