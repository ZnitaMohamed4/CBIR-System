/**
 * In a Dockerized production environment, we use a relative path '/api'.
 * The Nginx container (port 3000) will catch these requests and proxy 
 * them to the Python backend container (port 5000).
 */
const API_BASE_URL = '/api';

class ApiService {
  /**
   * Helper to handle fetch responses
   */
  async _handleResponse(response, errorMessage) {
    if (!response.ok) {
      let details = '';
      try {
        const errorData = await response.json();
        details = errorData.error || errorData.message || '';
      } catch (e) {
        details = response.statusText;
      }
      throw new Error(`${errorMessage}${details ? ': ' + details : ''}`);
    }
    return response.json();
  }

  // --- HEALTH & STATS ---

  async healthCheck() {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  }

  async getStats() {
    const response = await fetch(`${API_BASE_URL}/stats`);
    return response.json();
  }

  // --- IMAGE MANAGEMENT ---

  async uploadImage(file) {
    const formData = new FormData();
    formData.append('images', file);
    
    const response = await fetch(`${API_BASE_URL}/images/upload`, {
      method: 'POST',
      body: formData,
    });
    return this._handleResponse(response, 'Upload failed');
  }

  async uploadMultipleImages(files) {
    const formData = new FormData();
    files.forEach(file => formData.append('images', file));
    
    const response = await fetch(`${API_BASE_URL}/images/upload`, {
      method: 'POST',
      body: formData,
    });
    return this._handleResponse(response, 'Batch upload failed');
  }

  async getAllImages() {
    const response = await fetch(`${API_BASE_URL}/images`);
    return this._handleResponse(response, 'Failed to fetch images');
  }

  async getImage(imageId) {
    const response = await fetch(`${API_BASE_URL}/images/${imageId}`);
    return this._handleResponse(response, 'Failed to fetch image details');
  }

  async deleteImage(imageId) {
    const response = await fetch(`${API_BASE_URL}/images/${imageId}`, {
      method: 'DELETE',
    });
    return this._handleResponse(response, 'Delete failed');
  }

  async deleteImages(imageIds) {
    const response = await fetch(`${API_BASE_URL}/images`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: imageIds }),
    });
    return this._handleResponse(response, 'Batch delete failed');
  }

  // --- OBJECT DETECTION ---

  async detectObjects(imageId) {
    const response = await fetch(`${API_BASE_URL}/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_id: imageId }),
    });
    return this._handleResponse(response, 'Detection failed');
  }

  async detectObjectsBatch(imageIds) {
    const response = await fetch(`${API_BASE_URL}/detect/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: imageIds }),
    });
    return this._handleResponse(response, 'Batch detection failed');
  }

  // --- FEATURE EXTRACTION ---

  async extractFeatures(imageId, objectId) {
    const response = await fetch(`${API_BASE_URL}/features/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        image_id: imageId, 
        object_id: objectId 
      }),
    });
    return this._handleResponse(response, 'Feature extraction failed');
  }

  async extractFeaturesBatch(imageIds) {
    const response = await fetch(`${API_BASE_URL}/features/extract/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_ids: imageIds }),
    });
    return this._handleResponse(response, 'Batch extraction failed');
  }

  async getFeatures(imageId, objectId) {
    const response = await fetch(`${API_BASE_URL}/features/${imageId}/${objectId}`);
    return this._handleResponse(response, 'Failed to get features');
  }

  // --- SIMILARITY SEARCH ---

  async searchSimilar(queryImageId, queryObjectId, topK = 10, weights = null) {
    const response = await fetch(`${API_BASE_URL}/search/similar`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query_image_id: queryImageId,
        query_object_id: queryObjectId,
        top_k: topK,
        weights: weights,
      }),
    });
    return this._handleResponse(response, 'Search failed');
  }

  // --- TRANSFORMATIONS ---

  async transformImage(imageId, transformType, params) {
    const response = await fetch(`${API_BASE_URL}/images/${imageId}/transform`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        transform_type: transformType,
        params: params
      }),
    });
    return this._handleResponse(response, 'Transformation failed');
  }

  // --- URL HELPERS ---

  getImageUrl(filename) {
    if (!filename) return '';
    return `${API_BASE_URL}/images/file/${filename}`;
  }

  getDownloadUrl(imageId) {
    return `${API_BASE_URL}/images/download/${imageId}`;
  }

  async downloadImage(imageId, filename) {
    const response = await fetch(this.getDownloadUrl(imageId));
    if (!response.ok) throw new Error('Download failed');
    
    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || `image_${imageId}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }
}

export default new ApiService();