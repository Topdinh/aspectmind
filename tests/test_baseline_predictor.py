from aspectmind.inference.baseline_predictor import BaselinePredictor


def main():
    print("Loading baseline predictor...")
    predictor = BaselinePredictor("runs/baseline")

    texts = [
        "Pin trâu, hiệu năng ổn nhưng giá hơi cao.",
        "Camera chụp đêm kém, thiết kế đẹp.",
        "Dịch vụ bảo hành tốt, nhân viên nhiệt tình.",
    ]

    for text in texts:
        print("\n==============================")
        print("TEXT:", text)

        pred = predictor.predict(text)
        proba = predictor.predict_proba(text)

        print("PRED (0/1):", pred)
        print("PROBA:", {k: round(v, 4) for k, v in proba.items()})


if __name__ == "__main__":
    main()
