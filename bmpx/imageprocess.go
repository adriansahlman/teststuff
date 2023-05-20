package bmpx

import (
	"encoding/binary"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"sync"

	"github.com/disintegration/imaging"
)

type Header struct {
	Config       image.Config
	BitsPerPixel int
	TopDown      bool
	AllowAlpha   bool
	HeaderBytes  []byte
	ImageOffset  uint32
}

// bmpDecodeHeader was shamelessly copied from 'x/image/bmp' and edited for the
// usecase in this repo. Unlike the stdlib implementation, the header and
// palette bytes are retained so that they can be re-written to cropped images.
func DecodeHeader(r io.Reader) (res Header, err error) {
	readUint16 := func(b []byte) uint16 {
		return uint16(b[0]) | uint16(b[1])<<8
	}
	readUint32 := func(b []byte) uint32 {
		return uint32(
			b[0],
		) | uint32(
			b[1],
		)<<8 | uint32(
			b[2],
		)<<16 | uint32(
			b[3],
		)<<24
	}

	// We only support those BMP images with one of the following DIB headers:
	// - BITMAPINFOHEADER (40 bytes)
	// - BITMAPV4HEADER (108 bytes)
	// - BITMAPV5HEADER (124 bytes)
	const (
		fileHeaderLen   = 14
		infoHeaderLen   = 40
		v4InfoHeaderLen = 108
		v5InfoHeaderLen = 124
	)
	var empty Header
	var b [2048]byte
	if _, err := io.ReadFull(r, b[:fileHeaderLen+4]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return empty, err
	}
	if string(b[:2]) != "BM" {
		return empty, errors.New("bmp: invalid format")
	}
	offset := readUint32(b[10:14])
	res.ImageOffset = offset
	res.HeaderBytes = b[:offset]
	infoLen := readUint32(b[14:18])
	if infoLen != infoHeaderLen && infoLen != v4InfoHeaderLen &&
		infoLen != v5InfoHeaderLen {
		return empty, errors.New("unsupported")
	}
	if _, err := io.ReadFull(r, b[fileHeaderLen+4:fileHeaderLen+infoLen]); err != nil {
		if err == io.EOF {
			err = io.ErrUnexpectedEOF
		}
		return empty, err
	}
	width := int(int32(readUint32(b[18:22])))
	height := int(int32(readUint32(b[22:26])))
	if height < 0 {
		height, res.TopDown = -height, true
	}
	if width < 0 || height < 0 {
		return empty, errors.New("unsupported")
	}
	// We only support 1 plane and 8, 24 or 32 bits per pixel and no
	// compression.
	planes, bpp, compression := readUint16(
		b[26:28],
	), readUint16(
		b[28:30],
	), readUint32(
		b[30:34],
	)
	// if compression is set to BI_BITFIELDS, but the bitmask is set to the default bitmask
	// that would be used if compression was set to 0, we can continue as if compression was 0
	if compression == 3 && infoLen > infoHeaderLen &&
		readUint32(b[54:58]) == 0xff0000 && readUint32(b[58:62]) == 0xff00 &&
		readUint32(b[62:66]) == 0xff && readUint32(b[66:70]) == 0xff000000 {
		compression = 0
	}
	if planes != 1 || compression != 0 {
		return empty, errors.New("unsupported")
	}
	switch bpp {
	case 8:
		if offset != fileHeaderLen+infoLen+256*4 {
			return empty, errors.New("unsupported")
		}
		pre := fileHeaderLen + int(infoLen)
		_, err = io.ReadFull(r, b[pre:pre+256*4])
		if err != nil {
			return empty, err
		}
		pcm := make(color.Palette, 256)
		for i := range pcm {
			// BMP images are stored in BGR order rather than RGB order.
			// Every 4th byte is padding.
			pcm[i] = color.RGBA{b[pre+4*i+2], b[pre+4*i+1], b[pre+4*i+0], 0xFF}
		}
		res.Config = image.Config{ColorModel: pcm, Width: width, Height: height}
		res.BitsPerPixel = 8
		res.AllowAlpha = false
		return res, nil
	case 24:
		if offset != fileHeaderLen+infoLen {
			return empty, errors.New("unsupported")
		}
		res.Config = image.Config{
			ColorModel: color.RGBAModel,
			Width:      width,
			Height:     height,
		}
		res.BitsPerPixel = 24
		res.AllowAlpha = false
		return res, nil
	case 32:
		if offset != fileHeaderLen+infoLen {
			return empty, errors.New("unsupported")
		}
		// 32 bits per pixel is possibly RGBX (X is padding) or RGBA (A is
		// alpha transparency). However, for BMP images, "Alpha is a
		// poorly-documented and inconsistently-used feature" says
		// https://source.chromium.org/chromium/chromium/src/+/bc0a792d7ebc587190d1a62ccddba10abeea274b:third_party/blink/renderer/platform/image-decoders/bmp/bmp_image_reader.cc;l=621
		//
		// That goes on to say "BITMAPV3HEADER+ have an alpha bitmask in the
		// info header... so we respect it at all times... [For earlier
		// (smaller) headers we] ignore alpha in Windows V3 BMPs except inside
		// ICO files".
		//
		// "Ignore" means to always set alpha to 0xFF (fully opaque):
		// https://source.chromium.org/chromium/chromium/src/+/bc0a792d7ebc587190d1a62ccddba10abeea274b:third_party/blink/renderer/platform/image-decoders/bmp/bmp_image_reader.h;l=272
		//
		// Confusingly, "Windows V3" does not correspond to BITMAPV3HEADER, but
		// instead corresponds to the earlier (smaller) BITMAPINFOHEADER:
		// https://source.chromium.org/chromium/chromium/src/+/bc0a792d7ebc587190d1a62ccddba10abeea274b:third_party/blink/renderer/platform/image-decoders/bmp/bmp_image_reader.cc;l=258
		//
		// This Go package does not support ICO files and the (infoLen >
		// infoHeaderLen) condition distinguishes BITMAPINFOHEADER (40 bytes)
		// vs later (larger) headers.
		res.AllowAlpha = infoLen > infoHeaderLen
		res.Config = image.Config{
			ColorModel: color.RGBAModel,
			Width:      width,
			Height:     height,
		}
		res.BitsPerPixel = 32
		return res, nil
	}
	return empty, errors.New("unsupported")
}

// Crop crops the provided region of the BMP found in the input stream to the
// output stream.
//
// The input BMP must be bottom-up, no alpha, and uncompressed.
//
// Thanks to the simplicity of the BMP format, crop uses a very small amount of
// memory (~8KiB).
//
// If src is an io.ReadSeeker, then the cropper will seek to skip pixels that
// are outside the cropping region.
//
// Cropping complexity scales primarily with number of cropped rows, not
// columns. Depending on the data and number of crops, it may make sense to
// rotate the image accordingly.
func Crop(src io.Reader, dst io.Writer, region image.Rectangle) error {
	// Load BMP header bytes and significant content
	hdr, err := DecodeHeader(src)
	if err != nil {
		return err
	}
	if hdr.TopDown {
		return errors.New(".BMP: topDown not supported")
	}
	if hdr.AllowAlpha {
		return errors.New(".BMP: allowAlpha not supported")
	}

	// Find / validate crop area
	dim := image.Rect(0, 0, hdr.Config.Width, hdr.Config.Height)
	region = dim.Intersect(region)
	if region.Empty() {
		return errors.New("crop area empty or out of bounds")
	}

	// Create updated BMP header with crop dimensions
	totalSize := (hdr.BitsPerPixel/8)*(region.Dx()*region.Dy()) + len(
		hdr.HeaderBytes,
	)
	width := region.Dx()
	height := region.Dy()
	binary.LittleEndian.PutUint32(hdr.HeaderBytes[2:6], uint32(totalSize))
	binary.LittleEndian.PutUint32(hdr.HeaderBytes[18:22], uint32(width))
	binary.LittleEndian.PutUint32(hdr.HeaderBytes[22:26], uint32(height))
	_, err = dst.Write(hdr.HeaderBytes)
	if err != nil {
		return err
	}

	bytesPerPixel := hdr.BitsPerPixel / 8

	// Seek if possible, otherwise copy to discard
	var seek func(off int) (n int64, err error)
	if s, ok := src.(io.Seeker); ok {
		seek = func(off int) (n int64, err error) {
			return s.Seek(int64(off), io.SeekCurrent)
		}
	} else {
		seek = func(off int) (n int64, err error) {
			return io.CopyN(io.Discard, src, int64(off))
		}
	}

	byteWidth := func(pixels, bitsPerPixel int) int {
		return ((pixels*bitsPerPixel + 31) / 32) * 4
	}

	// Skip uncropped last rows (recall: bmp is bottom-up in this case)
	rowBytes := byteWidth(hdr.BitsPerPixel, hdr.Config.Width)
	skipBytes := rowBytes * (hdr.Config.Height - region.Max.Y)
	if _, err := seek(skipBytes); err != nil {
		return err
	}

	// Now within cropping region in terms of y
	// There are some nuances to be aware of: each BMP pixel row is padded to be
	// 4-byte aligned. This means that there may be extra bytes that are empty
	// on each row that is being read, and that padding may need to be added to
	// the row that is being written.
	left := bytesPerPixel * region.Min.X
	mid := region.Dx() * bytesPerPixel
	right := rowBytes - (mid + left)
	wantWidth := byteWidth(hdr.BitsPerPixel, region.Dx())
	padding := make([]byte, wantWidth-mid)

	for dy := 1; dy <= region.Dy(); dy++ {
		// Skip left
		_, err := seek(left)
		if err != nil {
			return err
		}

		// Write middle part with padding
		n, err := io.CopyN(dst, src, int64(mid))
		if err != nil || n != int64(mid) {
			return err
		}
		_, err = dst.Write(padding)
		if err != nil {
			return err
		}

		// Skip right
		_, err = seek(right)
		if err != nil {
			return err
		}
	}

	return nil
}

// Resize a BMP image as a stream. Holds the smallest
// amount of pixels possible in memory. Amount of pixels
// held in memory is equal to the width of the image
// multiplied by the height of the resampling filter.
// Each pixel takes up 4 * 8 bytes.
//
// Parts of this code are taken from or inspired by
// https://github.com/disintegration/imaging/blob/24d954dc01266ac1e8ba74cbe5e632c87fb0b38a/resize.go
func Resize(
	src io.Reader,
	dst io.Writer,
	width, height int,
	opts ...ResizeOption,
) error {
	if width <= 0 {
		return errors.New("resized width must be a positive value")
	}
	if height <= 0 {
		return errors.New("resized height must be a positive value")
	}
	// Set up default options
	o := resizeOptions{
		filter: imaging.Lanczos,
		pChunk: 64,
		pLimit: 4,
	}
	// Apply user option overrides
	for i := range opts {
		opts[i].apply(&o)
	}
	if err := o.validate(); err != nil {
		return err
	}

	// Decode BMP header of input
	hdr, err := DecodeHeader(src)
	if err != nil {
		return err
	}

	// TODO: fix 8-bit support
	if hdr.BitsPerPixel == 8 {
		return errors.New("8-bit pixels currently not supported")
	}

	// Validate format is supported
	if hdr.AllowAlpha {
		return errors.New(".BMP: allowAlpha not supported")
	}

	var decodePixels func(input []byte, pixels []uint8)
	// defaults to 24-bit encoding
	encodePixels := func(pixels []uint8, output []byte) {
		var locIn, locOut int
		pixelCount := len(pixels) / 4
		for i := 0; i < pixelCount; i++ {
			locIn = i * 4
			locOut = i * 3
			output[locOut] = pixels[locIn+2]
			output[locOut+1] = pixels[locIn+1]
			output[locOut+2] = pixels[locIn]
		}
	}

	bytesPerPixelIn := hdr.BitsPerPixel / 8
	bytesPerPixelOut := bytesPerPixelIn
	switch hdr.BitsPerPixel {
	case 8:
		// input is 8-bit pixels output is 24-bit pixels
		bytesPerPixelOut = 24 / 8
		srcPalette := hdr.Config.ColorModel.(color.Palette)
		palette := make([]color.NRGBA, len(srcPalette))
		for i := range srcPalette {
			palette[i] = color.NRGBAModel.Convert(srcPalette[i]).(color.NRGBA)
		}
		decodePixels = func(input []byte, pixels []uint8) {
			var loc int
			var c color.NRGBA
			for i, pI := range input {
				c = palette[pI]
				loc = i * 4
				pixels[loc] = c.R
				pixels[loc+1] = c.G
				pixels[loc+2] = c.B
				pixels[loc+3] = c.A
			}
		}
	case 24:
		decodePixels = func(input []byte, pixels []uint8) {
			var locIn, locOut int
			pixelCount := len(input) / 3
			for i := 0; i < pixelCount; i++ {
				locIn = i * 3
				locOut = i * 4
				pixels[locOut+0] = input[locIn+2]
				pixels[locOut+1] = input[locIn+1]
				pixels[locOut+2] = input[locIn+0]
				pixels[locOut+3] = 0xFF
			}
		}
	case 32:
		decodePixels = func(input []byte, pixels []uint8) {
			copy(pixels, input)
			pixelCount := len(input) / 4
			var loc int
			for i := 0; i < pixelCount; i++ {
				loc = i * 4
				pixels[loc+0], pixels[loc+2] = pixels[loc+2], pixels[loc+0]
			}
		}
		encodePixels = func(pixels []uint8, output []byte) {
			copy(output, pixels)
			pixelCount := len(pixels) / 4
			var loc int
			for i := 0; i < pixelCount; i++ {
				loc = i * 4
				output[loc+0], output[loc+2] = output[loc+2], output[loc+0]
			}
		}
	default:
		return fmt.Errorf(
			"unsupported number of bits per pixel: %d",
			hdr.BitsPerPixel,
		)
	}
	// Update BMP header with new image dimensions
	binary.LittleEndian.PutUint32(
		hdr.HeaderBytes[2:6],
		uint32(bytesPerPixelOut*(width*height)+len(hdr.HeaderBytes)),
	)
	binary.LittleEndian.PutUint32(hdr.HeaderBytes[18:22], uint32(width))
	binary.LittleEndian.PutUint32(hdr.HeaderBytes[22:26], uint32(height))

	// Write BMP header
	if _, err = dst.Write(hdr.HeaderBytes); err != nil {
		return err
	}
	widthIn, heightIn := hdr.Config.Width, hdr.Config.Height
	// Check for no-op
	if width == widthIn && height == heightIn {
		// No resize, copy entire image and return
		_, err = io.Copy(dst, src)
		return err
	}

	floatToByte := func(x float64) uint8 {
		v := int64(x + 0.5)
		if v > 255 {
			return 255
		}
		if v > 0 {
			return uint8(v)
		}
		return 0
	}

	type chunk struct {
		start int
		stop  int
	}

	// parallel processes the data in separate goroutines.
	parallel := func(start, stop int, fn func(<-chan chunk)) {
		if stop <= start {
			return
		}
		count := (stop-start-1)/o.pChunk + 1

		if o.pLimit > count {
			o.pLimit = count
		}
		min := func(a, b int) int {
			if a < b {
				return a
			}
			return b
		}
		c := make(chan chunk, count)
		for i := start; i < stop; i += o.pChunk {
			c <- chunk{start: i, stop: min(i+o.pChunk, stop)}
		}
		close(c)

		if o.pLimit <= 1 {
			fn(c)
			return
		}

		var wg sync.WaitGroup
		for i := 0; i < o.pLimit; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				fn(c)
			}()
		}
		wg.Wait()
	}

	// Pre-calculate pixel weights
	var weightsX, weightsY [][]indexWeight
	if width != widthIn {
		weightsX, _ = computeWeights(width, widthIn, o.filter)
	}
	kernelSizeY := 1
	if height != heightIn {
		weightsY, kernelSizeY = computeWeights(
			height,
			heightIn,
			o.filter,
		)
	}

	// Holds the input pixels
	inputPixelBuf := make([]uint8, widthIn*4)

	// Holds the horizontally resized rows
	intermPixelBuf := make([]uint8, kernelSizeY*width*4)

	// Holds the bytes for a row of input pixels
	// plus any additional padding
	rowBytesBufIn := make(
		[]byte,
		bytesPerPixelIn*widthIn+getNumPaddingBytes(
			widthIn,
			bytesPerPixelIn,
		),
	)
	// Holds the bytes for a row of output pixels
	// plus any additional padding
	rowBytesBufOut := make(
		[]byte,
		bytesPerPixelOut*width+getNumPaddingBytes(
			width,
			bytesPerPixelOut,
		),
	)

	y0In, y1In := heightIn-1, -1
	y0Out, y1Out := height-1, -1
	yDelta := -1
	if hdr.TopDown {
		y0In, y1In = 0, heightIn
		y0Out, y1Out = 0, height
		yDelta = 1
	}

	// Tracks the current input row
	yIn := y0In
	// Tracks the next output row to resize and write
	yOut := y0Out

	canWriteCurrentOutputRow := func() bool {
		if hdr.TopDown {
			return weightsY[yOut][len(weightsY[yOut])-1].index <= yIn
		}
		return weightsY[yOut][0].index >= yIn
	}

	// For each input row
	for ; yIn != y1In; yIn += yDelta {
		if _, err = io.ReadFull(src, rowBytesBufIn); err != nil {
			return fmt.Errorf(
				"failed to read %d row bytes for y=%d: %w",
				len(rowBytesBufIn),
				yIn,
				err,
			)
		}
		// Decode pixels of current input row
		parallel(0, widthIn, func(chunks <-chan chunk) {
			for c := range chunks {
				decodePixels(
					rowBytesBufIn[c.start*bytesPerPixelIn:c.stop*bytesPerPixelIn],
					inputPixelBuf[c.start*4:c.stop*4],
				)
			}
		})

		// Horizontally resize row and store in buffer
		if weightsX == nil {
			// No horizontal resizing, simply copy input row
			loc := yIn % kernelSizeY * width * 4
			copy(
				intermPixelBuf[loc:loc+width*4],
				inputPixelBuf,
			)
		} else {
			// Resize the row
			parallel(0, width, func(chunks <-chan chunk) {
				var rgbaWeightedOut [4]float64
				var w indexWeight
				var x, loc int
				var aw, aInv float64
				for c := range chunks {
					for x = c.start; x < c.stop; x++ {
						rgbaWeightedOut = [4]float64{}
						for _, w = range weightsX[x] {
							loc = w.index * 4
							aw = float64(inputPixelBuf[loc+3]) * w.weight
							rgbaWeightedOut[0] += float64(inputPixelBuf[loc+0]) * aw
							rgbaWeightedOut[1] += float64(inputPixelBuf[loc+1]) * aw
							rgbaWeightedOut[2] += float64(inputPixelBuf[loc+2]) * aw
							rgbaWeightedOut[3] += aw
						}
						loc = (yIn%kernelSizeY*width + x) * 4
						intermPixelBuf[loc+0] = 0
						intermPixelBuf[loc+1] = 0
						intermPixelBuf[loc+2] = 0
						intermPixelBuf[loc+3] = 0
						if rgbaWeightedOut[3] != 0 {
							aInv = 1 / rgbaWeightedOut[3]
							intermPixelBuf[loc+0] = floatToByte(rgbaWeightedOut[0] * aInv)
							intermPixelBuf[loc+1] = floatToByte(rgbaWeightedOut[1] * aInv)
							intermPixelBuf[loc+2] = floatToByte(rgbaWeightedOut[2] * aInv)
							intermPixelBuf[loc+3] = floatToByte(rgbaWeightedOut[3])
						}
					}
				}
			})
		}
		// No vertical resize, encode and
		// then write the row immediately
		if weightsY == nil {
			parallel(0, width, func(chunks <-chan chunk) {
				for c := range chunks {
					loc := (yIn%kernelSizeY*width + c.start) * 4
					encodePixels(
						intermPixelBuf[loc:loc+(c.stop-c.start)*4],
						rowBytesBufOut[c.start*bytesPerPixelOut:c.stop*bytesPerPixelOut],
					)
				}
			})
			if _, err = dst.Write(rowBytesBufOut); err != nil {
				return fmt.Errorf(
					"failed to write %d row bytes for y=%d: %w",
					len(rowBytesBufOut),
					yOut,
					err,
				)
			}
			continue
		}

		// If enough rows have been horizontally resized for the
		// current output row, vertically resize and write it.
		// Continue this action until more input rows are required
		// or all output rows have been written.
		for ; yOut != y1Out && canWriteCurrentOutputRow(); yOut += yDelta {
			parallel(0, width, func(chunks <-chan chunk) {
				var rgbaWOut [4]float64
				var rgbaOut [4]uint8
				var w indexWeight
				var x, loc int
				var aw, aInv float64
				for c := range chunks {
					for x = c.start; x < c.stop; x++ {
						rgbaWOut, rgbaOut = [4]float64{}, [4]uint8{}
						for _, w = range weightsY[yOut] {
							loc = (w.index%kernelSizeY*width + x) * 4
							aw = float64(intermPixelBuf[loc+3]) * w.weight
							rgbaWOut[0] += float64(intermPixelBuf[loc+0]) * aw
							rgbaWOut[1] += float64(intermPixelBuf[loc+1]) * aw
							rgbaWOut[2] += float64(intermPixelBuf[loc+2]) * aw
							rgbaWOut[3] += aw
						}
						rgbaOut[0] = 0
						rgbaOut[1] = 0
						rgbaOut[2] = 0
						rgbaOut[3] = 0
						if rgbaWOut[3] != 0 {
							aInv = 1 / rgbaWOut[3]
							rgbaOut[0] = floatToByte(rgbaWOut[0] * aInv)
							rgbaOut[1] = floatToByte(rgbaWOut[1] * aInv)
							rgbaOut[2] = floatToByte(rgbaWOut[2] * aInv)
							rgbaOut[3] = floatToByte(rgbaWOut[3])
						}
						encodePixels(
							rgbaOut[:],
							rowBytesBufOut[x*bytesPerPixelOut:(x+1)*bytesPerPixelOut],
						)
					}
				}
			})
			if _, err = dst.Write(rowBytesBufOut); err != nil {
				return fmt.Errorf(
					"failed to write %d row bytes for y=%d: %w",
					len(rowBytesBufOut),
					yOut,
					err,
				)
			}
		}
	}
	return nil
}

type resizeOptions struct {
	filter imaging.ResampleFilter
	// size of work batches processed
	// by workers (go routines)
	pChunk int
	// parallel limit
	pLimit int
}

func (o *resizeOptions) validate() error {
	if o.filter.Support <= 0 {
		return errors.New(
			"unsupported filter, filter.Support must be larger than 0",
		)
	}
	if o.pChunk <= 0 {
		return errors.New(
			"invalid value for parallel batch size, must be greater than 0",
		)
	}
	if o.pLimit < 0 {
		return errors.New(
			"invalid value for parallel limit, must be greater or equal to 0",
		)
	}
	return nil
}

type ResizeOption interface {
	apply(*resizeOptions)
}

type resizeOptionFunc func(*resizeOptions)

func (f resizeOptionFunc) apply(opts *resizeOptions) {
	f(opts)
}

// Resampling filter used for resizing.
func WithResizeFilter(filter imaging.ResampleFilter) ResizeOption {
	return resizeOptionFunc(func(opts *resizeOptions) {
		opts.filter = filter
	})
}

// Maximum number of parallel workers (go routines).
func WithResizeParallelLimit(limit int) ResizeOption {
	return resizeOptionFunc(func(opts *resizeOptions) {
		opts.pLimit = limit
	})
}

// Number of pixels in each job that the workers
// (go routines) take on.
func WithResizeParallelBatchSize(chunk int) ResizeOption {
	return resizeOptionFunc(func(opts *resizeOptions) {
		opts.pChunk = chunk
	})
}

func getNumPaddingBytes(width, bytesPerPixel int) int {
	if n := 4 + -bytesPerPixel*width%4; n != 4 {
		return n
	}
	return 0
}

// Code below adapted from
// https://github.com/disintegration/imaging/blob/24d954dc01266ac1e8ba74cbe5e632c87fb0b38a/resize.go

type indexWeight struct {
	index  int
	weight float64
}

// Returns the normalized kernel weights
// for a single dimension and the kernel size.
// For each output index there is a set
// of weights for corresponding input values.
func computeWeights(
	dstSize, srcSize int,
	filter imaging.ResampleFilter,
) ([][]indexWeight, int) {
	du := float64(srcSize) / float64(dstSize)
	scale := du
	if scale < 1.0 {
		scale = 1.0
	}
	ru := math.Ceil(scale * filter.Support)

	out := make([][]indexWeight, dstSize)
	tmp := make([]indexWeight, 0, dstSize*int(ru+2)*2)

	var begin, end, u, kernelSize int
	var fu, sum, w float64
	for v := 0; v < dstSize; v++ {
		fu = (float64(v)+0.5)*du - 0.5

		begin = int(math.Ceil(fu - ru))
		if begin < 0 {
			begin = 0
		}
		end = int(math.Floor(fu + ru))
		if end > srcSize-1 {
			end = srcSize - 1
		}

		sum = 0
		for u = begin; u <= end; u++ {
			if w = filter.Kernel((float64(u) - fu) / scale); w != 0 {
				sum += w
				tmp = append(tmp, indexWeight{index: u, weight: w})
			}
		}
		if sum != 0 {
			for i := range tmp {
				tmp[i].weight /= sum
			}
		}
		// theres probably a more clever way
		// of calculating kernel size but meh
		if len(tmp) > kernelSize {
			kernelSize = len(tmp)
		}
		out[v] = tmp
		tmp = tmp[len(tmp):]
	}
	return out, kernelSize
}
