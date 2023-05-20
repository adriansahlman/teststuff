package bmpx_test

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"io"
	"math"
	"testing"

	"github.com/adriansahlman/teststuff/bmpx"
	"github.com/disintegration/imaging"
	"github.com/stretchr/testify/require"
	"golang.org/x/image/bmp"
)

var filters = []struct {
	name   string
	filter imaging.ResampleFilter
}{
	{"Box", imaging.Box},
	{"Linear", imaging.Linear},
	{"Hermite", imaging.Hermite},
	{"MitchellNetravali", imaging.MitchellNetravali},
	{"CatmullRom", imaging.CatmullRom},
	{"BSpline", imaging.BSpline},
	{"Gaussian", imaging.Gaussian},
	{"Bartlett", imaging.Bartlett},
	{"Hann", imaging.Hann},
	{"Hamming", imaging.Hamming},
	{"Blackman", imaging.Blackman},
	{"Welch", imaging.Welch},
	{"Cosine", imaging.Cosine},
}

func TestResize(t *testing.T) {
	buf := &bytes.Buffer{}
	var expected, actual image.Image
	var ok bool
	// can not test 32-bit with alpha since
	// go's bmp lib does not support it.
	// 8-bit needs more work before it will
	// function correctly.
	// TODO: Add 8-bit to testing.
	for _, bpp := range []int{24} {
		srcImg := generateImage(100, 175, bpp)
		buf.Reset()
		err := bmp.Encode(buf, srcImg)
		require.NoError(t, err)
		src := append([]byte(nil), buf.Bytes()...)
		for _, filter := range filters {
			for _, width := range []int{175, 100, 10} {
				for _, height := range []int{175, 100, 10} {
					name := fmt.Sprintf(
						"%s/%d/%dx%d",
						filter.name,
						bpp,
						width,
						height,
					)
					ok = t.Run(name, func(t *testing.T) {
						expected = imaging.Resize(
							srcImg,
							width,
							height,
							filter.filter,
						)
						buf.Reset()
						err = bmpx.Resize(
							bytes.NewReader(src),
							buf,
							width,
							height,
							bmpx.WithResizeFilter(filter.filter),
						)
						require.NoError(t, err)
						actual, err = bmp.Decode(bytes.NewReader(buf.Bytes()))
						require.NoError(t, err)
						requireEqualImages(t, expected, actual)
					})
					if !ok {
						t.SkipNow()
					}
				}
			}
		}
	}
}

func BenchmarkResize(b *testing.B) {
	sizes := []int{16, 256, 1024, 4096, 8192}
	buf := &bytes.Buffer{}
	for _, filter := range []struct {
		name   string
		filter imaging.ResampleFilter
	}{
		{"Box", imaging.Box},
		{"Lanczos", imaging.Lanczos},
	} {
		for i, srcSize := range sizes[1:] {
			srcImg := generateImage(srcSize, srcSize, 24)
			buf.Reset()
			err := bmp.Encode(buf, srcImg)
			require.NoError(b, err)
			src := append([]byte(nil), buf.Bytes()...)
			for _, width := range sizes[:i+1] {
				if width > srcSize {
					continue
				}
				for _, height := range sizes[:i+1] {
					if width == srcSize && height == srcSize {
						continue
					}
					name := fmt.Sprintf(
						"%s/%dx%d/%dx%d",
						filter.name,
						srcSize,
						srcSize,
						width,
						height,
					)
					ok := b.Run(name+"/Memory", func(b *testing.B) {
						for i := 0; i < b.N; i++ {
							img, err := bmp.Decode(bytes.NewReader(src))
							require.NoError(b, err)
							img = imaging.Resize(
								img,
								width,
								height,
								filter.filter,
							)
							err = bmp.Encode(io.Discard, img)
							require.NoError(b, err)
						}
					})
					if !ok {
						b.SkipNow()
					}
					b.Run(name+"/Stream", func(b *testing.B) {
						for i := 0; i < b.N; i++ {
							err := bmpx.Resize(
								bytes.NewReader(src),
								io.Discard,
								width,
								height,
								bmpx.WithResizeFilter(
									filter.filter,
								),
							)
							require.NoError(b, err)
						}
					})
				}
			}
		}
	}
}

func requireEqualImages(
	t *testing.T,
	expected, actual image.Image,
) {
	type RGBA struct{ R, G, B, A uint32 }
	require.Equal(t, expected.Bounds(), actual.Bounds(), "image sizes")
	for y := expected.Bounds().Min.Y; y < expected.Bounds().Max.Y; y++ {
		for x := expected.Bounds().Min.X; x < expected.Bounds().Max.X; x++ {
			eR, eG, eB, eA := expected.At(x, y).RGBA()
			aR, aG, aB, aA := actual.At(x, y).RGBA()
			eRgba := RGBA{eR, eG, eB, eA}
			aRgba := RGBA{aR, aG, aB, aA}
			require.Equal(
				t,
				eRgba,
				aRgba,
				fmt.Sprintf("Pixel at (%d, %d)", x, y),
			)
		}
	}
}

func generateImage(width, height, bitsPerPixel int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	var c color.NRGBA
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			c = color.NRGBA{
				R: uint8(math.Round(float64(y) / float64(height-1) * 255)),
				G: uint8(math.Round(float64(x) / float64(width-1) * 255)),
				B: uint8(
					math.Round(
						float64(x+y) / float64(width+height-2) * 255,
					),
				),
				A: 255 - uint8(
					math.Round(
						float64(x+y)/float64(width+height-2)*255/2,
					),
				),
			}
			if bitsPerPixel == 24 || bitsPerPixel == 8 {
				c.A = 0xFF
			}
			img.Set(x, y, c)
		}
	}
	switch bitsPerPixel {
	case 8:
		palettedImage := image.NewPaletted(img.Bounds(), palette.Plan9)
		draw.Draw(
			palettedImage,
			palettedImage.Rect,
			img,
			img.Bounds().Min,
			draw.Over,
		)
		return palettedImage
	case 24, 32:
		return img
	default:
		panic(bitsPerPixel)
	}
}
