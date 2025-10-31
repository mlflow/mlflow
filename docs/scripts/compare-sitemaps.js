var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
import * as fs from 'fs';
import { XMLParser } from 'fast-xml-parser';
import fetch from 'node-fetch';
function readSitemap(input) {
    return __awaiter(this, void 0, void 0, function () {
        var res, responseBody;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!/^https?:\/\//.test(input)) return [3 /*break*/, 5];
                    return [4 /*yield*/, fetch(input)];
                case 1:
                    res = _a.sent();
                    if (!!res.ok) return [3 /*break*/, 3];
                    return [4 /*yield*/, res.text()];
                case 2:
                    responseBody = _a.sent();
                    throw new Error("Failed to fetch ".concat(input, ": ").concat(res.status, " ").concat(res.statusText, ". Response body: ").concat(responseBody));
                case 3: return [4 /*yield*/, res.text()];
                case 4: return [2 /*return*/, _a.sent()];
                case 5: return [4 /*yield*/, fs.promises.readFile(input, 'utf8')];
                case 6: return [2 /*return*/, _a.sent()];
            }
        });
    });
}
function normalizePath(url) {
    var idx = url.indexOf('/latest/');
    return idx >= 0 ? url.slice(idx + 'latest/'.length) : url;
}
function parseSitemap(input) {
    return __awaiter(this, void 0, void 0, function () {
        var xml, parser, parsed, urlset, urlMap, _i, urlset_1, loc, normalized;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, readSitemap(input)];
                case 1:
                    xml = _a.sent();
                    parser = new XMLParser();
                    parsed = parser.parse(xml);
                    urlset = parsed.urlset.url;
                    urlMap = new Map();
                    for (_i = 0, urlset_1 = urlset; _i < urlset_1.length; _i++) {
                        loc = urlset_1[_i].loc;
                        normalized = normalizePath(loc);
                        urlMap.set(normalized, loc);
                    }
                    return [2 /*return*/, urlMap];
            }
        });
    });
}
function compareSitemaps(mapA, mapB) {
    var onlyInA = [];
    var onlyInB = [];
    var inBoth = [];
    for (var _i = 0, mapA_1 = mapA; _i < mapA_1.length; _i++) {
        var _a = mapA_1[_i], url = _a[0], _ = _a[1];
        if (!mapB.has(url)) {
            onlyInA.push(url);
        }
        else {
            inBoth.push(url);
        }
    }
    for (var _b = 0, _c = mapB.keys(); _b < _c.length; _b++) {
        var url = _c[_b];
        if (!mapA.has(url)) {
            onlyInB.push(url);
        }
    }
    return { onlyInA: onlyInA, onlyInB: onlyInB, inBoth: inBoth };
}
(function () { return __awaiter(void 0, void 0, void 0, function () {
    var fileA, fileB, _a, mapA, mapB, _b, onlyInA, onlyInB, inBoth;
    return __generator(this, function (_c) {
        switch (_c.label) {
            case 0:
                fileA = process.argv[2];
                fileB = process.argv[3];
                if (!fileA || !fileB) {
                    console.error('Usage: tsx compare-sitemaps.ts <fileA|urlA> <fileB|urlB>');
                    process.exit(1);
                }
                return [4 /*yield*/, Promise.all([parseSitemap(fileA), parseSitemap(fileB)])];
            case 1:
                _a = _c.sent(), mapA = _a[0], mapB = _a[1];
                _b = compareSitemaps(mapA, mapB), onlyInA = _b.onlyInA, onlyInB = _b.onlyInB, inBoth = _b.inBoth;
                console.log("URLs in both: ".concat(inBoth.length));
                console.log("Only in ".concat(fileA, ": ").concat(onlyInA.length));
                onlyInA.forEach(function (url) { return console.log("  ".concat(url)); });
                console.log("Only in ".concat(fileB, ": ").concat(onlyInB.length));
                onlyInB.forEach(function (url) { return console.log("  ".concat(url)); });
                return [2 /*return*/];
        }
    });
}); })();
