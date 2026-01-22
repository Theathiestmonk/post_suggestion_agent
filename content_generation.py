# content_generator.py
"""
Content Generation Module

This module handles the actual generation of captions and images using the prompts
created by the RL system.

- Caption generation: OpenAI GPT-4o-mini
- Image generation: Gemini 2.5 Flash Image Preview
"""
import os
import json
import base64
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from db import SupabaseImageManager, supabase
import requests
from PIL import Image
import io
try:
    # Try to use the newer google.genai package
    import google.genai as genai
    USE_NEW_PACKAGE = True
except ImportError:
    # Fall back to deprecated package
    import google.generativeai as genai
    USE_NEW_PACKAGE = False

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class ContentGenerator:
    """Handles caption and image generation using AI models."""

    def __init__(self):
        """Initialize the content generator with API clients."""
        self.openai_client = None
        self.gemini_client = None
        self.image_manager = None

        if OPENAI_API_KEY:
            self.openai_client = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=OPENAI_API_KEY,
                temperature=0.7
            ) 

        if GEMINI_API_KEY:
            if USE_NEW_PACKAGE:
                # New google.genai package
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            else:
                # Legacy google.generativeai package
                genai.configure(api_key=GEMINI_API_KEY)
                try:
                    # Use the requested model for image generation
                    self.gemini_client = genai.GenerativeModel('gemini-2.5-flash-image-preview')
                except Exception as e:
                    print(f"Warning: Could not initialize gemini-2.5-flash-image-preview ({e}), trying fallback")
                    try:
                        self.gemini_client = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')
                    except Exception:
                        print("Warning: Could not initialize any Gemini image generation model")
                        self.gemini_client = None
        else:
            self.gemini_client = None

        # Initialize Supabase image manager
        try:
            self.image_manager = SupabaseImageManager()
        except Exception as e:
            print(f"Warning: Could not initialize Supabase image manager: {e}")
            self.image_manager = None

    def fetch_business_logo(self, business_id: str) -> Optional[str]:
        """
        Fetch logo URL from business profile in database.

        Args:
            business_id: The business ID to fetch logo for

        Returns:
            Logo URL string or None if not found
        """
        try:
            res = supabase.table("profiles").select("logo_url").eq("id", business_id).execute()
            if res.data and len(res.data) > 0:
                logo_url = res.data[0].get("logo_url")
                if logo_url:
                    print(f"📥 Found logo URL for business {business_id}: {logo_url}")
                    return logo_url
            print(f"📭 No logo found for business {business_id}")
            return None
        except Exception as e:
            print(f"❌ Error fetching logo for business {business_id}: {e}")
            return None

    def generate_caption(self, caption_prompt: str) -> str:
        """
        Generate a caption using OpenAI GPT-4o-mini.

        Args:
            caption_prompt: The prompt to generate the caption from

        Returns:
            Plain text caption with hashtags
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not configured")

        try:
            response = self.openai_client.invoke([
                HumanMessage(content=caption_prompt)
            ])

            # Return the response content directly (should include hashtags as per prompt)
            return response.content.strip()

        except Exception as e:
            raise RuntimeError(f"Failed to generate caption: {e}")

    def generate_image(self, image_prompt: str, logo_url: str = None, business_id: str = None) -> str:
        """
        Generate an image using Gemini with optional logo overlay.

        Args:
            image_prompt: The prompt to generate the image from
            logo_url: Optional direct logo URL to overlay
            business_id: Optional business ID to fetch logo from database

        Returns:
            Base64 encoded image data URL or placeholder URL
        """
        if not self.gemini_client:
            raise ValueError("Gemini API key not configured")

        # Handle logo functionality
        final_logo_url = logo_url
        if business_id and not logo_url:
            # Fetch logo from database if business_id provided but no direct logo_url
            final_logo_url = self.fetch_business_logo(business_id)

        logo_image = None
        if final_logo_url:
            try:
                print(f"📥 Downloading logo from: {final_logo_url}")
                # Download logo image
                response = requests.get(final_logo_url, timeout=10)
                response.raise_for_status()

                # Convert to PIL Image for Gemini
                logo_image = Image.open(io.BytesIO(response.content))
                print("✅ Logo downloaded and processed successfully")

                # Modify prompt to include logo placement instructions
                logo_instructions = """

                IMPORTANT LOGO PLACEMENT REQUIREMENTS:
                - Place the provided logo image in the BOTTOM RIGHT CORNER of the generated image
                - Make the logo clearly visible and prominent
                - Scale the logo to take MAXIMUM 6-10% of the total image area
                - Ensure the logo maintains its aspect ratio and quality
                - Position it professionally without overlapping important content
                - The logo should enhance the professional appearance of the image

                """
                image_prompt += logo_instructions
                print("🎨 Added logo placement instructions to prompt")

            except Exception as e:
                print(f"⚠️ Failed to download/process logo: {e}")
                logo_image = None

        try:
            if USE_NEW_PACKAGE:
                # New google.genai package API
                if logo_image:
                    # Include logo image in the request
                    from google.genai import types
                    contents = [
                        image_prompt,
                        logo_image  # PIL Image object
                    ]
                else:
                    contents = image_prompt

                response = self.gemini_client.models.generate_content(
                    model='gemini-2.5-flash-image-preview',
                    contents=contents
                )

                # Handle new package response format
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        image_data = None
                        for part in candidate.content.parts:
                            if hasattr(part, 'inline_data') and part.inline_data is not None:
                                inline_data = part.inline_data
                                if hasattr(inline_data, 'data') and inline_data.data is not None:
                                    image_data = inline_data.data
                                    break  # Found the image data, exit the loop
                                elif hasattr(inline_data, 'blob') and inline_data.blob is not None:
                                    # Try 'blob' attribute instead
                                    image_data = inline_data.blob
                                    break  # Found the image data, exit the loop

                        # Now check if we found image data
                        if image_data is not None:
                            # Upload to Supabase storage and get public URL
                            try:
                                if self.image_manager:
                                    public_url = self.image_manager.save_image(
                                        image_data=image_data,
                                        filename="generated_image.png",
                                        folder="generated"
                                    )
                                    return public_url
                                else:
                                    raise RuntimeError("Image manager not initialized")
                            except Exception as upload_error:
                                print(f"Failed to upload to Supabase: {upload_error}")
                                # Fallback to base64 data URL
                                base64_image = base64.b64encode(image_data).decode('utf-8')
                                return f"data:image/png;base64,{base64_image}"
                # If no image generated, return a placeholder
                return f"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

            else:
                # Legacy google.generativeai package API
                if logo_image:
                    # For legacy API, we need to convert PIL image to the format expected
                    import google.generativeai as genai
                    # Convert PIL image to genai.Image format
                    logo_genai = genai.Image.from_pil(logo_image)
                    contents = [image_prompt, logo_genai]
                    response = self.gemini_client.generate_content(contents)
                else:
                    response = self.gemini_client.generate_content(image_prompt)

                # Get the generated image data
                if response.candidates and response.candidates[0].content.parts:
                    image_part = response.candidates[0].content.parts[0]

                    if hasattr(image_part, 'inline_data'):
                        # Get image data
                        image_data = image_part.inline_data.data

                        # Upload to Supabase storage and get public URL
                        try:
                            if self.image_manager:
                                public_url = self.image_manager.save_image(
                                    image_data=image_data,
                                    filename="generated_image.png",
                                    folder="generated"
                                )
                                return public_url
                            else:
                                raise RuntimeError("Image manager not initialized")
                        except Exception as upload_error:
                            print(f"Failed to upload to Supabase: {upload_error}")
                            # Fallback to base64 data URL
                            mime_type = image_part.inline_data.mime_type
                            base64_image = base64.b64encode(image_data).decode('utf-8')
                            return f"data:{mime_type};base64,{base64_image}"

                raise RuntimeError("No image generated in response")

        except Exception as e:
            # Raise clear error instead of returning placeholder
            error_msg = f"Image generation failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)


    def generate_content(self, caption_prompt: str, image_prompt: str, business_context: dict = None, logo_url: str = None, business_id: str = None) -> Dict[str, Any]:
        """
        Generate both caption and image from their respective prompts.

        Args:
            caption_prompt: Prompt for caption generation
            image_prompt: Prompt for image generation
            business_context: Business profile data dictionary for structured context formatting

        Returns:
            Dict with 'caption' and 'image_url' keys
        """
        try:
            caption = self.generate_caption(caption_prompt)

            # Format and append structured business context to image prompt if provided
            if business_context:
                formatted_context = format_business_context(business_context)
                image_prompt += f"\n\nBusiness Context: \n\n{formatted_context}"
                print(f"📋 Appended structured business context to image prompt")

            image_url = self.generate_image(image_prompt, logo_url, business_id)

            return {
                "caption": caption,
                "image_url": image_url,
                "status": "success"
            }

        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }


def format_business_context(profile_data: dict) -> str:
    """
    Format business profile data into structured context for image generation LLMs.

    Args:
        profile_data: Business profile dictionary from database

    Returns:
        Formatted business context string
    """
    context_parts = []

    # Core business information
    if profile_data.get("business_name"):
        context_parts.append(f"Business Name: {profile_data['business_name']}")

    if profile_data.get("industries"):
        industries = ", ".join(profile_data["industries"]) if isinstance(profile_data["industries"], list) else profile_data["industries"]
        context_parts.append(f"Industry: {industries}")

    if profile_data.get("business_types"):
        business_types = ", ".join(profile_data["business_types"]) if isinstance(profile_data["business_types"], list) else profile_data["business_types"]
        context_parts.append(f"Business Type: {business_types}")

    # Description and value proposition
    if profile_data.get("business_description"):
        context_parts.append(f"Description: {profile_data['business_description']}")

    if profile_data.get("unique_value_proposition"):
        context_parts.append(f"Unique Value Proposition: {profile_data['unique_value_proposition']}")

    # Brand and audience information
    if profile_data.get("brand_voice"):
        context_parts.append(f"Brand Voice: {profile_data['brand_voice']}")

    if profile_data.get("target_audience"):
        context_parts.append(f"Target Audience: {profile_data['target_audience']}")

    # Visual preferences - Always include color information
    primary_color = profile_data.get("primary_color", "Not specified")
    context_parts.append(f"Primary Color: {primary_color}")

    secondary_color = profile_data.get("secondary_color", "Not specified")
    context_parts.append(f"Secondary Color: {secondary_color}")

    # Join all parts with proper formatting
    return "\n\n".join(context_parts)


# Convenience functions for direct use
def generate_caption(caption_prompt: str) -> str:
    """Generate a caption from a prompt."""
    generator = ContentGenerator()
    return generator.generate_caption(caption_prompt)


def generate_image(image_prompt: str) -> str:
    """Generate an image from a prompt."""
    generator = ContentGenerator()
    return generator.generate_image(image_prompt)


def generate_content(caption_prompt: str, image_prompt: str, business_context: dict = None, logo_url: str = None, business_id: str = None) -> Dict[str, Any]:
    """Generate both caption and image from their prompts with optional logo overlay."""
    generator = ContentGenerator()
    return generator.generate_content(caption_prompt, image_prompt, business_context, logo_url, business_id)


def test_logo_overlay():
    """Test the logo overlay functionality"""
    print("🧪 Testing Logo Overlay Functionality in ATSN_RL_FINAL")
    print("=" * 60)

    # Test image prompt
    image_prompt = "Create a powerful professional image showing a modern office workspace with a laptop, coffee mug, and business documents on a clean desk. Make it look corporate and successful."

    # Test business ID (you would need to replace this with an actual business ID that has a logo)
    test_business_id = "your-business-id-here"  # Replace with actual business ID

    generator = ContentGenerator()

    try:
        print("1️⃣ Testing logo fetch from database...")
        logo_url = generator.fetch_business_logo(test_business_id)
        if logo_url:
            print(f"   ✅ Found logo: {logo_url}")
        else:
            print("   ⚠️ No logo found, will use placeholder")
            logo_url = "https://picsum.photos/200/100?random=logo"

        print("\n2️⃣ Generating image WITH logo overlay...")
        image_url_with_logo = generator.generate_image(image_prompt, logo_url, test_business_id)
        print(f"   ✅ Image with logo generated")
        print(f"   📍 URL: {image_url_with_logo[:100]}...")

        print("\n3️⃣ Testing full content generation with logo...")
        caption_prompt = "Write an engaging Instagram caption about productivity tips for entrepreneurs, including relevant hashtags."

        result_with_logo = generator.generate_content(
            caption_prompt,
            image_prompt,
            logo_url=logo_url,
            business_id=test_business_id
        )
        print(f"   📝 Caption: {result_with_logo['caption'][:100]}...")
        print(f"   🖼️ Image with logo: {result_with_logo['image_url'][:100]}...")

        print("\n🎉 Logo overlay functionality test completed!")
        print("\n📋 Test Results:")
        print(f"   • Logo URL: {logo_url}")
        print(f"   • Image with logo: {image_url_with_logo}")
        print(f"   • Content with logo: {result_with_logo['image_url']}")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_logo_overlay()
    # Example usage
    caption_prompt = "Write an engaging Instagram caption about productivity tips for entrepreneurs, including relevant hashtags."
    image_prompt = "Create a professional image showing a laptop with coffee and notebooks, modern office setting."

    generator = ContentGenerator()

    try:
        result = generator.generate_content(caption_prompt, image_prompt)
        print("Generated Content:")
        print(f"Caption: {result['caption']}")
        print(f"Image URL: {result['image_url'][:100]}...")
    except Exception as e:
        print(f"Error: {e}")