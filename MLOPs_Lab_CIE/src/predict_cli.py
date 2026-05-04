import argparse
import joblib
import pandas as pd

MODEL_PATH = "models/model.pkl"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stride_length_cm", type=float, required=True)
    parser.add_argument("--ground_contact_ms", type=float, required=True)
    parser.add_argument("--hip_drop_degrees", type=float, required=True)
    parser.add_argument("--fatigue_index", type=float, required=True)

    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([{
        "stride_length_cm": args.stride_length_cm,
        "ground_contact_ms": args.ground_contact_ms,
        "hip_drop_degrees": args.hip_drop_degrees,
        "fatigue_index": args.fatigue_index
    }])

    prediction = model.predict(input_df)[0]

    print(round(float(prediction), 3))

if __name__ == "__main__":
    main()