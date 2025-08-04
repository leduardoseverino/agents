# GitHub OAuth Setup for AG2 Enhanced Documentation Flow

## Quick Setup Guide

### 1. Create GitHub OAuth Application

#### For GitHub.com:
1. Go to https://github.com/settings/applications/new
2. Fill in the application details:
   - **Application name**: `AG2 Enhanced Documentation Flow`
   - **Homepage URL**: `http://localhost:8000`
   - **Authorization callback URL**: `http://localhost:8000/api/auth/callback`
3. Click "Register application"
4. Copy the **Client ID** and **Client Secret**

#### For GitHub Enterprise:
1. Go to `https://your-github-enterprise.com/settings/applications/new`
2. Fill in the same application details as above
3. Register and copy the credentials

### 2. Set Environment Variables

Create a `.env` file in the `GDM` directory with your OAuth credentials:

```bash
# .env file
GITHUB_CLIENT_ID=your_github_oauth_app_client_id_here
GITHUB_CLIENT_SECRET=your_github_oauth_app_client_secret_here
```

### 3. Run the Application

```bash
cd GDM
python start_web.py
```

## Features

✅ **GitHub.com Support**: Authenticate with your regular GitHub account  
✅ **GitHub Enterprise Support**: Works with GitHub Enterprise Server  
✅ **Repository Access**: Browse and select from all your accessible repositories  
✅ **Secure OAuth Flow**: Industry-standard OAuth 2.0 authentication  
✅ **Session Management**: Secure session handling with automatic logout  

## Usage

1. Open your browser to `http://localhost:8000`
2. Select either GitHub.com or GitHub Enterprise
3. If using Enterprise, enter your server URL
4. Click "Login with GitHub"
5. Authorize the application in the popup window
6. Select a repository from your list
7. Configure analysis settings
8. Start the enhanced documentation analysis!

## Troubleshooting

**Issue**: "Authentication failed"  
**Solution**: Verify your OAuth app credentials and callback URL

**Issue**: "No repositories found"  
**Solution**: Ensure your GitHub account has repository access

**Issue**: "Enterprise server not found"  
**Solution**: Verify the Enterprise server URL is correct and accessible

## Production Deployment

For production deployment, update the OAuth app settings:
- **Homepage URL**: `https://yourdomain.com`
- **Authorization callback URL**: `https://yourdomain.com/api/auth/callback`

And update your environment variables accordingly.