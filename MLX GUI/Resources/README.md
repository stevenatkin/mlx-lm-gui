# Model List Customization

The app loads models from two locations:

1. **Bundled Models** (`popular_models.json`): Default models included with the app
2. **User Custom Models** (`custom_models.json`): User-specific models stored in Application Support

## Adding Custom Models

To add your own models to the selection list:

1. Create a file named `custom_models.json` in:
   ```
   ~/Library/Application Support/MLX Training Studio/custom_models.json
   ```

2. Use the same JSON format as `popular_models.json`:
   ```json
   {
     "models": [
       {
         "identifier": "your-org/your-model-name",
         "displayName": "Your Model Display Name"
       },
       {
         "identifier": "another-org/another-model",
         "displayName": "Another Model"
       }
     ]
   }
   ```

3. The app will automatically merge your custom models with the default models when it loads.

## Format

- `identifier`: The Hugging Face model identifier (e.g., `mlx-community/Qwen2.5-7B-Instruct`)
- `displayName`: The name shown in the dropdown menu (e.g., `Qwen2.5 7B Instruct (4bit)`)

## Notes

- Custom models are merged with bundled models, avoiding duplicates by identifier
- If a custom model has the same identifier as a bundled model, the bundled model takes precedence
- Changes to `custom_models.json` require restarting the app to take effect
- You can always enter any Hugging Face model identifier manually in the text field, even if it's not in the list

