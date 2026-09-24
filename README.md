# Dry-run, 5 cycles, 15-minute interval
python kraken_knapsack_trader.py --interval 900 --max-cycles 5

# Live (only after testing in dry-run)
export KRAKEN_API_KEY="your_public_key"
export KRAKEN_API_SECRET="your_base64_private_key"
export ENABLE_LIVE_TRADING=YES_I_UNDERSTAND

python kraken_knapsack_trader.py \
    --live \
    --confirm-live \
    --interval 900 \
    --max-cycles 0
