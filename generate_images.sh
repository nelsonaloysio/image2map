#!/usr/bin/env bash
#
# Generates images to train the SOM Image2Map model
# by rotating a black line drawn on a white background.
#
# Requires ImageMagick installed.

USAGE="""Usage:
  $(basename $0) -s SAMPLES -d DIMENSIONS
                 [--image-ext FORMAT] [--output-images PATH]

Arguments:
  -h, --help                display usage information
  -s, --samples SAMPLES     number of images to generate (required)
  -d, --dim DIMENSIONS      dimension, i.e., width/height (required)
  --image-ext FORMAT        extension format (optional; default: 'png')
  --output-images PATH      name or path of folder to create (optional)

Example to generate 100 images with 16x16 dimensions:
  $(basename $0) -s 100 -d 16x16
"""

# Displays usage information and exit.
usage() { echo -n "$USAGE" && exit 0; }

while [[ $# -gt 0 ]]; do
    case $1 in
        ""|-h|--help)
            usage
            ;;
        -s|--samples)
            SAMPLES="$2"
            shift
            ;;
        -d|--dim)
            W_DIM="$(echo "$2" | cut -f1 -d'x')"
            H_DIM="$(echo "$2" | cut -f2 -d'x')"
            shift
            ;;
        --image-ext)
            IMAGE_EXT="$2"
            shift
            ;;
        --output-images)
            OUTPUT_PATH="$2"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Verify ImageMagick is installed.
[ -z "$(command -v convert)" ] &&
echo "Error: binary 'convert' not found. Is ImageMagick installed?" &&
exit 1

# Exit if required parameters are missing.
[ -z "$SAMPLES" ] && echo "Error: missing required parameter: 'SAMPLES'." && exit 1
[ -z "$W_DIM" ] && echo "Error: missing required parameter: 'DIMENSIONS'." && exit 1

# Fallback to default parameters.
[ -z "$IMAGE_EXT" ] && IMAGE_EXT=png
[ -z "$OUTPUT_PATH" ] && OUTPUT_PATH="data/images_${SAMPLES}_${W_DIM}x${H_DIM}_$IMAGE_EXT"

# Warn if folder already exists.
[ -d "$OUTPUT_PATH" ] && echo "Warning: folder '$OUTPUT_PATH' already exists."

# Create folder.
mkdir -p "$OUTPUT_PATH"

# Get angle variation, height, and width to generate lines.
A=$( bc -l <<< "scale=2; 180/$SAMPLES")
H=$( echo $(( 1 + $(( $H_DIM / 10 )) )) | cut -f1 -d\. )
W=$(( $W_DIM - $H ))

# Generate sample line image (in black color).
convert -size ${W}x${H} xc:black "${OUTPUT_PATH}/image_0.${IMAGE_EXT}"
echo "Generating $SAMPLES ${W_DIM}x${H_DIM} ${IMAGE_EXT^^} images (line: ${W}x${H}, rotating $A degrees)..."

a=0; i=0
while [ "$(bc -l <<< "$a<180")" -eq 1 ]; do
    i=$(($i + 1))

    # Generate rotated images with white background.
    convert +dither \
            -background white \
            -rotate $a \
            -extent ${W_DIM}x${H_DIM} \
            -gravity center \
            -monochrome \
            -colors 2 \
            -colorspace gray \
            "${OUTPUT_PATH}/image_0.${IMAGE_EXT}" \
            "${OUTPUT_PATH}/image_${i}.${IMAGE_EXT}"

    # Compute angle variation for next iteration.
    a=$(bc -l <<< "$a+$A")
    echo -n "$i "
done
echo "OK"

# Remove sample line image.
rm -f "${OUTPUT_PATH}/image_0.${IMAGE_EXT}"
