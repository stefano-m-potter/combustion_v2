src="$NOBACKUP/cnn_mapping/Russia/local_predict_all_filter_mod_active_missing"
dst="$NOBACKUP/cnn_mapping/Russia/local_predict_all_filter_landsat_mod_active"

# Loop through each year in the source folder
for year in $(ls "$src"); do
  if [ -d "$src/$year" ]; then
    echo "Moving files for year $year..."
    mv "$src/$year"/* "$dst/$year"/
  fi
done

echo "✅ All missing files moved into local_predict_all_filter."
