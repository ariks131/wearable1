file_list=("/home/cml3/Wearable-pi/data_logger.py" "/home/cml3/Wearable-pi/run_edge_model_with_raw_old_data.py")

COUNT="1"
while true; do
echo "Processing data $COUNT..." &

python3 "/home/cml3/Wearable-pi/data_logger.py"
python3 "/home/cml3/Wearable-pi/run_edge_model_with_raw_old_data.py"

python3 "/home/cml3/Wearable-pi/display_result.py" &
wait
done
echo "Exiting."
