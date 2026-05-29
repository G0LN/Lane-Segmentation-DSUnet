import os
import zipfile
import sys

def zip_project(output_filename="dsunet_code.zip"):
    # Root directory of the project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Folders and file extensions to exclude
    exclude_dirs = {
        '.git', '.venv', 'data', 'checkpoints', 'logs', 
        '__pycache__', '.ipynb_checkpoints', 'results'
    }
    exclude_extensions = {'.zip', '.pth', '.onnx', '.data'}
    
    print(f"=== PACKAGING DSUNET CODE FOR KAGGLE ===")
    print(f"Project root: {root_dir}")
    print(f"Excluding heavy folders: {list(exclude_dirs)}")
    print(f"Excluding extensions: {list(exclude_extensions)}")
    
    zip_path = os.path.join(root_dir, output_filename)
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    count = 0
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Modify dirs in-place to prevent os.walk from entering excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # Check file extension
                _, ext = os.path.splitext(file)
                if ext.lower() in exclude_extensions:
                    continue
                if file == output_filename:
                    continue
                    
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, root_dir)
                
                zipf.write(full_path, relative_path)
                count += 1
                
    print("-" * 50)
    print(f"[SUCCESS] Packaged {count} source files successfully!")
    print(f"Zip file created at: {zip_path}")
    print(f"-> Size: {os.path.getsize(zip_path) / 1024:.2f} KB")
    print(f"You can now upload this ZIP file directly to Kaggle as a Dataset!")
    print("=" * 50)

if __name__ == "__main__":
    zip_project()
