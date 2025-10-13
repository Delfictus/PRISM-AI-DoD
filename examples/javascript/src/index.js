/**
 * PRISM-AI JavaScript Client Library
 *
 * Official JavaScript/Node.js SDK for the PRISM-AI REST API.
 */

const PrismClient = require('./client');
const exceptions = require('./exceptions');
const models = require('./models');

module.exports = {
  PrismClient,
  ...exceptions,
  ...models,
};
