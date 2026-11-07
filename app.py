
# ============================================================
# CYCLEGAN STYLE STUDIO
# Hugging Face Spaces + ZeroGPU + Gradio 6
# ============================================================

# IMPORTANT: Import spaces BEFORE torch.
import spaces

import os
import uuid
import torch
import gradio as gr
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from torch import nn


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "CycleGAN Style Studio"

HORSE_MODEL = "horse2zebra_G.pth"
MONET_MODEL = "monet2photo_G.pth"

OUTPUT_DIR = "/tmp/cyclegan_gallery"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 55)
print("CYCLEGAN STYLE STUDIO STARTING")
print("=" * 55)


# ============================================================
# 2. RESNET BLOCK
# ============================================================

class ResnetBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv_block = nn.Sequential(

            nn.ReflectionPad2d(1),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3
            ),

            nn.InstanceNorm2d(
                channels,
                affine=False,
                track_running_stats=True
            ),

            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3
            ),

            nn.InstanceNorm2d(
                channels,
                affine=False,
                track_running_stats=True
            )
        )

    def forward(self, x):

        return x + self.conv_block(x)


# ============================================================
# 3. RESNET GENERATOR ARCHITECTURE
# ============================================================

class ResnetGenerator(nn.Module):

    def __init__(self):
        super().__init__()

        layers = [

            # Initial convolution
            nn.ReflectionPad2d(3),

            nn.Conv2d(
                3, 64,
                kernel_size=7
            ),

            nn.InstanceNorm2d(
                64,
                affine=False,
                track_running_stats=True
            ),

            nn.ReLU(inplace=True),

            # Downsampling 1
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(
                128,
                affine=False,
                track_running_stats=True
            ),

            nn.ReLU(inplace=True),

            # Downsampling 2
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.InstanceNorm2d(
                256,
                affine=False,
                track_running_stats=True
            ),

            nn.ReLU(inplace=True)
        ]

        # Nine residual blocks
        for _ in range(9):

            layers.append(
                ResnetBlock(256)
            )

        # Upsampling
        layers.extend([

            nn.ConvTranspose2d(
                256, 128,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.InstanceNorm2d(
                128,
                affine=False,
                track_running_stats=True
            ),

            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),

            nn.InstanceNorm2d(
                64,
                affine=False,
                track_running_stats=True
            ),

            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(3),

            nn.Conv2d(
                64, 3,
                kernel_size=7
            ),

            nn.Tanh()
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)


# ============================================================
# 4. LOAD PRETRAINED MODELS ON CPU
# ============================================================

def load_model(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    print(f"Loading model: {path}")

    model = ResnetGenerator()

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        state,
        strict=True
    )

    # Keep models on CPU until GPU inference is requested.
    model.to("cpu")
    model.eval()

    print("Model loaded successfully!")

    return model


print("\nLoading CycleGAN models...")

horse_model = load_model(HORSE_MODEL)

monet_model = load_model(MONET_MODEL)

print("\nBoth models loaded successfully!")

print("Models are ready for ZeroGPU inference.")

print("=" * 55)


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    if image is None:

        raise ValueError(
            "Please upload an image first."
        )

    original = image.convert("RGB")

    resized = original.resize(
        (256, 256),
        Image.Resampling.LANCZOS
    )

    array = np.asarray(
        resized,
        dtype=np.float32
    )

    # Normalize image between -1 and 1
    array = (array / 127.5) - 1.0

    tensor = torch.from_numpy(array)

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    return original, tensor


# ============================================================
# 6. IMAGE GENERATION
# ============================================================

def generate_image(image, direction, strength):

    original, tensor = preprocess_image(image)

    if direction == "Horse → Zebra":

        model = horse_model

    elif direction == "Monet Painting → Photograph":

        model = monet_model

    else:

        raise ValueError(
            "Unsupported translation direction."
        )

    try:

        # Move selected model and input to GPU
        model.to("cuda")

        tensor = tensor.to("cuda")

        with torch.inference_mode():

            generated = model(tensor)

        # Convert generated tensor to CPU
        generated = generated.squeeze(0).cpu()

        generated = (generated + 1.0) / 2.0

        generated = generated.clamp(0, 1)

        generated = generated.permute(
            1, 2, 0
        ).numpy()

        generated = (
            generated * 255
        ).astype(np.uint8)

        generated_image = Image.fromarray(
            generated
        )

        # Restore original image dimensions
        generated_image = generated_image.resize(
            original.size,
            Image.Resampling.LANCZOS
        )

        # Blend original and generated images
        result = Image.blend(
            original,
            generated_image,
            float(strength)
        )

        return result

    finally:

        # Release GPU memory
        model.to("cpu")

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# ============================================================
# 7. WATERMARK
# ============================================================

def add_watermark(image):

    image = image.convert("RGB")

    draw = ImageDraw.Draw(image)

    text = "CycleGAN Style Studio | AI Generated"

    try:

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            max(12, image.width // 45)
        )

    except OSError:

        font = ImageFont.load_default()

    x = 10

    y = max(5, image.height - 35)

    draw.text(
        (x, y),
        text,
        fill="white",
        font=font,
        stroke_width=2,
        stroke_fill="black"
    )

    return image


# ============================================================
# 8. PLAIN-ENGLISH REQUEST ROUTING
# ============================================================

def select_direction(request, selected_direction):

    prompt = (request or "").lower()

    # Horse and Zebra requests
    if "horse" in prompt or "zebra" in prompt:

        return "Horse → Zebra"

    # Monet painting to photograph requests
    if any(
        word in prompt
        for word in [
            "monet",
            "painting",
            "photograph",
            "photo"
        ]
    ):

        return "Monet Painting → Photograph"

    return selected_direction


# ============================================================
# 9. TRANSLATION FUNCTION
# ============================================================

@spaces.GPU(duration=120)
def translate(image, request, direction, strength):

    try:

        if image is None:

            raise ValueError(
                "Please upload an image."
            )

        print("Starting image transformation...")

        # Select translation direction
        direction = select_direction(
            request,
            direction
        )

        # Generate image
        result = generate_image(
            image,
            direction,
            strength
        )

        # Apply watermark
        result = add_watermark(result)

        # Unique output filename
        filename = os.path.join(
            OUTPUT_DIR,
            f"image_{uuid.uuid4().hex[:8]}.png"
        )

        # Save generated image
        result.save(filename)

        print("Image saved:", filename)

        print("Transformation completed!")

        return (
            result,
            [(result, direction)],
            filename,
            f"SUCCESS! {direction} transformation completed."
        )

    except Exception as e:

        print("GENERATION ERROR:", str(e))

        return (
            None,
            [],
            None,
            f"ERROR: {str(e)}"
        )


# ============================================================
# 10. GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(
    title=APP_TITLE
) as demo:

    gr.Markdown(
        """
        # 🎨 CycleGAN Style Studio

        ### AI-Powered Image Translation Platform

        Transform your images using pretrained CycleGAN models.

        **Available transformations:**

        - 🐎 Horse → Zebra
        - 🎨 Monet Painting → Photograph

        Upload an image, describe your transformation,
        select the model and generate your AI image.
        """
    )

    with gr.Row():

        # LEFT COLUMN
        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload Your Image"
            )

            prompt = gr.Textbox(
                label="Plain-English Request",
                placeholder="Example: Convert this horse into a zebra"
            )

            direction = gr.Dropdown(
                choices=[
                    "Horse → Zebra",
                    "Monet Painting → Photograph"
                ],
                value="Horse → Zebra",
                label="Translation Direction"
            )

            strength = gr.Slider(
                minimum=0,
                maximum=1,
                value=1,
                step=0.05,
                label="Transformation Strength"
            )

            generate_btn = gr.Button(
                "✨ Generate Image",
                variant="primary"
            )

        # RIGHT COLUMN
        with gr.Column():

            output_image = gr.Image(
                label="Transformed Image"
            )

            download_file = gr.File(
                label="Download Generated Image"
            )

            status = gr.Textbox(
                label="Generation Status",
                interactive=False
            )

    # IMAGE GALLERY
    gallery = gr.Gallery(
        label="Generated Image Gallery",
        columns=3,
        height=300
    )

    # GENERATE BUTTON
    generate_btn.click(

        fn=translate,

        inputs=[
            input_image,
            prompt,
            direction,
            strength
        ],

        outputs=[
            output_image,
            gallery,
            download_file,
            status
        ]
    )

    # PROJECT INFORMATION
    gr.Markdown(
        """
        ---

        ## About CycleGAN

        CycleGAN is a generative adversarial network
        used for unpaired image-to-image translation.

        It learns mappings between two visual domains
        using adversarial and cycle-consistency losses.

        ## Features

        - Pretrained CycleGAN models
        - Image-to-image translation
        - Transformation strength controller
        - Generated image gallery
        - Downloadable results
        - AI-generated watermark

        ## Copyright Notice

        Generated images are provided for educational
        and research purposes.

        This application is not affiliated with
        the original artists or copyright holders.

        **Developed for Generative AI – CA Mini Project.**
        """
    )


# ============================================================
# 11. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":

    print("\nStarting CycleGAN Style Studio...")

    demo.queue()

    demo.launch(

        server_name="0.0.0.0",

        server_port=7860,

        theme=gr.themes.Soft()

    )
