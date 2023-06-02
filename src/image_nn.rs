use image::GenericImageView;

pub fn load_image(
    path: &String,
    inputs: &mut Vec<Vec<f64>>,
    targets: &mut Vec<Vec<f64>>,
    width: &mut f32,
    height: &mut f32,
) {
    let img = image::open(path).expect("Failed to open image");

    let (w, h) = img.dimensions();

    *width = w as f32;
    *height = h as f32;

    for y in 0..h {
        for x in 0..w {
            let vec1 = vec![x as f64 / w as f64, y as f64 / h as f64];
            inputs.push(vec1);

            let pixel = img.get_pixel(x, y);

            let brightness = if pixel[0] > 0 { 1.0 } else { 0.0 };
            targets.push(vec![brightness]);
        }
    }
}
