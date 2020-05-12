const express = require('express');
const moment = require('moment');
const parser = require('body-parser');
const db = require('./models');
const app = express();

app.use(parser.json());

app.use(async (req, res, next) => {
    if (req.headers.authorization !== 'authToken') {
        return res
            .status(401)
            .send();
    }

    next();
});

//versions resource
app.get('/api/v1/versions', async (req, res) => {
    let versions = await db.versions
        .find_all();

    return res
        .status(200)
        .send(versions);
});

app.get('/api/v1/versions/:version_id', async (req, res) => {
    const version_id = req.params.version_id;

    let version = await db.versions
        .find_one({
            where: {
                version_id
            }
        });
    if (!version) {
        return res
            .status(404)
            .send();
    }

    return res
        .status(200)
        .send(version);
});

app.post('/api/v1/versions', async (req, res) => {
    const { name } = req.body;

    let version = await db.versions
        .create({
            name,
            created_at: moment().format()
        });

    return res
        .status(201)
        .send(version)
});

app.put('/api/v1/versions/:version_id', async (req, res) => {
    const version_id = req.params.version_id;
    const { name } = req.body;

    let version = await db.versions
        .find_one({
            where: {
                version_id
            }
        });

    if (!version) {
        return res
            .status(404)
            .send();
    }

    version.name = name;

    await version.save();

    return res
        .status(200)
        .send(version)
});

app.delete('api/v1/versions/:version_id', async (req, res) => {
    const version_id = req.params.version_id;

    let version = await db.versions
        .find_one({
            where: {
                version_id
            }
        });

    if (!version) {
        return res
            .status(404)
            .send();
    }

    await version.destroy();

    return res
        .status(204)
        .send();
});

